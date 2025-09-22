import cv2
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from threading import Thread
import queue

# ============================================================================
# Depth Dual-Camera Viewer with Obstacle Detection & Steering Guidance
# ----------------------------------------------------------------------------
# This script fuses two camera feeds, estimates depth, highlights detected
# obstacles, and overlays guidance for steering around them.
#
# The overall program flow is:
#   1. Configure tunable parameters (grid density, detection thresholds, etc.).
#   2. Start the depth-estimation model and camera capture pipelines.
#   3. Spawn background threads to keep frames and inference results flowing.
#   4. Within the main loop:
#        * Build a sparse depth cloud on a configurable grid.
#        * Detect obstacles by comparing depth values to global statistics.
#        * Render visualizations (depth colormap, cutoffs, detected points).
#        * Plan a clear horizontal route and draw steering guidance.
#   5. Press ESC to exit cleanly.
# ============================================================================

# ---------------------------------------------------------------------------
# 🔧 Tunable Parameters (grouped for quick tweaking)
# ---------------------------------------------------------------------------

# Grid sampling density for obstacle detection. Higher values sample more
# points but increase computation time. These values directly control the size
# of the sparse point cloud used for analysis.
GRID_ROWS = 25
GRID_COLS = 50

# Obstacle detection sensitivity. The model compares each sampled depth value
# against the global statistics of the entire sparse cloud. Tune these to make
# detection more or less aggressive.
DEPTH_DIFF_THRESHOLD = 5    # Minimum absolute difference from the mean depth.
STD_MULTIPLIER = 0.23       # Multiplier for the standard-deviation-based gate.

# Vertical cutoff boundaries (in pixels from the respective frame edges). Red
# obstacle markers will only appear between these lines. Adjust to ignore sky
# or the vehicle body. The same values are used for both cameras.
TOP_CUTOFF_PIXELS = 10
BOTTOM_CUTOFF_PIXELS = 54

# Horizontal route-planner smoothing. Values near 0 snap instantly to a new
# target gap; values near 1 move very slowly for a damped response.
BLUE_X_SMOOTHNESS = 0.7

# Width of the "pull" zone around the combined-image center. Obstacles inside
# this area directly influence the route planner. Obstacles outside only count
# if the pull zone already contains something.
PULL_INFLUENCE_RADIUS_PX = 120

# ---------------------------------------------------------------------------
# 📦 Runtime State (initialized here for clarity)
# ---------------------------------------------------------------------------

# Persistent estimate of the blue guidance circle's X-position (float for
# smooth interpolation).
blue_x = None

# Current steering command that gets printed on-screen each frame.
steer_control_x = None


def clamp(val, minn, maxn):
    """Clamp *val* to the inclusive range [minn, maxn]."""
    return max(min(val, maxn), minn)


# ---------------------------------------------------------------------------
# 🚀 Initialize the Depth-Estimation Pipeline
# ---------------------------------------------------------------------------
depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    device=0
)


# ---------------------------------------------------------------------------
# 🎥 Camera Capture Setup (Jetson CSI via GStreamer)
# ---------------------------------------------------------------------------
def make_gst(sensor_id):
    """Return a GStreamer pipeline string for the given CSI camera."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM),width=300,height=150,framerate=60/1 ! "
        "nvvidconv flip-method=2 ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink sync=false max-buffers=1 drop=true"
    )


# Open both camera feeds using the helper above.
cap0 = cv2.VideoCapture(make_gst(0), cv2.CAP_GSTREAMER)
cap1 = cv2.VideoCapture(make_gst(1), cv2.CAP_GSTREAMER)
if not cap0.isOpened() or not cap1.isOpened():
    raise RuntimeError("Failed to open one or both cameras.")

# Keep the buffer shallow so we always process the freshest frame.
cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ---------------------------------------------------------------------------
# 🧵 Threaded Queues for Camera Frames and Depth Inference
# ---------------------------------------------------------------------------
frame_q0, frame_q1, result_q = queue.Queue(1), queue.Queue(1), queue.Queue(1)
running = True


def grab(cam, q):
    """Continuously grab frames from *cam* and push the latest into *q*."""
    while running:
        ret, frame = cam.read()
        if ret:
            if q.full():
                q.get_nowait()  # discard stale frame to avoid blocking
            q.put(frame)


# Launch frame grabbers for both cameras.
Thread(target=grab, args=(cap0, frame_q0), daemon=True).start()
Thread(target=grab, args=(cap1, frame_q1), daemon=True).start()


def infer():
    """Fetch paired frames, run depth estimation, and queue the results."""
    while running:
        try:
            f0 = frame_q0.get(timeout=0.01)
            f1 = frame_q1.get(timeout=0.01)
        except queue.Empty:
            continue

        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in (f0, f1)]

        with torch.amp.autocast(device_type='cuda'), torch.no_grad():
            outs = depth_pipe(imgs)

        d0, d1 = [np.array(o['depth']) for o in outs]

        if result_q.full():
            result_q.get_nowait()
        result_q.put((f0, d0, f1, d1))


# Depth inference runs on its own background thread.
Thread(target=infer, daemon=True).start()


# ---------------------------------------------------------------------------
# 🔁 Main Visualization & Planning Loop
# ---------------------------------------------------------------------------
while True:
    try:
        frame0, depth0, frame1, depth1 = result_q.get(timeout=0.01)
    except queue.Empty:
        continue

    # Build sparse point cloud
    h0, w0 = depth0.shape
    h1, w1 = depth1.shape
    points = []
    for cam_idx, (depth, x_offset, w, h) in enumerate([
        (depth0, 0, w0, h0), (depth1, w0, w1, h1)
    ]):
        for c in range(GRID_COLS):
            for r in range(GRID_ROWS):
                px = int((c + 0.5) * w / GRID_COLS) + x_offset
                py = int((r + 0.5) * h / GRID_ROWS)
                z  = float(depth[int(py), int(px - x_offset)])
                points.append((px, py, z, cam_idx))
    cloud = np.array(points, dtype=[('x','f4'),('y','f4'),('z','f4'),('cam','i4')])

    # Obstacle detection thresholds
    zs = cloud['z']
    mean_z, std_z = zs.mean(), zs.std()
    thresh = max(DEPTH_DIFF_THRESHOLD, STD_MULTIPLIER * std_z)
    mask = (mean_z - zs) > thresh

    # Prepare visualizations
    norm0 = cv2.normalize(depth0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cmap0 = cv2.applyColorMap(cv2.resize(norm0, (frame0.shape[1], frame0.shape[0])), cv2.COLORMAP_MAGMA)
    norm1 = cv2.normalize(depth1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cmap1 = cv2.applyColorMap(cv2.resize(norm1, (frame1.shape[1], frame1.shape[0])), cv2.COLORMAP_MAGMA)

    # Draw cutoff lines on each camera
    for cmap, h, w in [(cmap0, frame0.shape[0], frame0.shape[1]), (cmap1, frame1.shape[0], frame1.shape[1])]:
        # Top cutoff line (ignore everything above)
        cv2.line(cmap, (0, TOP_CUTOFF_PIXELS), (w, TOP_CUTOFF_PIXELS), (0, 255, 0), 1)

        # Bottom cutoff line (ignore everything below)
        bottom_y = h - BOTTOM_CUTOFF_PIXELS
        cv2.line(cmap, (0, bottom_y), (w, bottom_y), (0, 255, 0), 1)

    # Draw obstacle points only within cutoffs
    for is_obst, pt in zip(mask, cloud):
        if not is_obst:
            continue
        px, py, cam = int(pt['x']), int(pt['y']), pt['cam']
        # check cutoffs
        if py < TOP_CUTOFF_PIXELS or py > ((frame0.shape[0] if cam == 0 else frame1.shape[0]) - BOTTOM_CUTOFF_PIXELS):
            continue
        # draw red dot
        target = cmap0 if cam == 0 else cmap1
        cv2.circle(target, (px - (0 if cam==0 else w0), py), 5, (0, 0, 255), -1)

    # Show combined view
    combined = np.hstack((cmap0, cmap1))
    
    
    
    # === 🟦 HORIZONTAL GAP ROUTE PLANNER ────────────────────────────────
    h_combined, w_combined = frame0.shape[0], frame0.shape[1] + frame1.shape[1]

    # vertical midpoint between the two green cutoff lines
    line_y = (TOP_CUTOFF_PIXELS + (h_combined - BOTTOM_CUTOFF_PIXELS)) // 2

    # combined-frame center X
    center_x = w_combined // 2

    # pull zone boundaries (clamped to image)
    zone_left = clamp(center_x - PULL_INFLUENCE_RADIUS_PX, 0, w_combined)
    zone_right = clamp(center_x + PULL_INFLUENCE_RADIUS_PX, 0, w_combined)

    # Collect red-dot X positions that fall between the cutoff green lines (both cams)
    red_xs_all = []
    red_xs_in_zone = []

    for is_obst, pt in zip(mask, cloud):
        if not is_obst:
            continue
        px, py, cam = int(pt['x']), int(pt['y']), pt['cam']
        frame_h = frame0.shape[0] if cam == 0 else frame1.shape[0]
        if py < TOP_CUTOFF_PIXELS or py > (frame_h - BOTTOM_CUTOFF_PIXELS):
            continue
        red_xs_all.append(px)
        if zone_left <= px <= zone_right:
            red_xs_in_zone.append(px)

    # Deduplicate & sort
    red_xs_all = sorted(set(red_xs_all))
    red_xs_in_zone = sorted(set(red_xs_in_zone))

    # Global blockers across the full width
    blockers_all = [0] + red_xs_all + [w_combined]

    def widest_gap_center(blockers, preferred_x):
        """
        Choose the center of the widest gap considering ALL blockers.
        Tie-breaker: pick the gap whose center is closest to preferred_x.
        """
        if len(blockers) < 2:
            return preferred_x
        best_width = -1
        best_dist = 1e18
        best_cx = preferred_x
        for left, right in zip(blockers[:-1], blockers[1:]):
            width = right - left
            cx = (left + right) // 2
            dist = abs(cx - preferred_x)
            if (width > best_width) or (width == best_width and dist < best_dist):
                best_width = width
                best_dist = dist
                best_cx = cx
        return int(best_cx)

    # Gating:
    # - No in-zone obstacles -> ignore outside, keep center
    # - Any in-zone obstacles -> plan using ALL obstacles
    if len(red_xs_in_zone) == 0:
        gap_cx = center_x
    else:
        gap_cx = widest_gap_center(blockers_all, preferred_x=center_x)

    # Smooth horizontal motion (keep as FLOAT; don't floor!)
    if blue_x is None:
        blue_x = float(gap_cx)
    else:
        blue_x = blue_x * BLUE_X_SMOOTHNESS + gap_cx * (1 - BLUE_X_SMOOTHNESS)

    # Round ONLY for rendering / display to avoid bias
    draw_x = int(round(blue_x))

    # Update "Steer Control" variable every loop
    steer_control_x = draw_x

    # Draw the guidance circle
    blue_pos = (draw_x, line_y)
    cv2.circle(combined, blue_pos, 12, (255, 0, 0), 3)

    # Display "Steer Control" on the left frame
    cv2.putText(
        combined,
        f"Steer Control: {steer_control_x}",
        (10, 30),  # top-left inside the left frame
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # Visualize the pull zone boundaries
    cv2.line(combined, (int(zone_left), 0), (int(zone_left), h_combined), (255, 0, 0), 1)
    cv2.line(combined, (int(zone_right), 0), (int(zone_right), h_combined), (255, 0, 0), 1)
    # ────────────────────────────────────────────────────────────────────


    
    
    
    
    cv2.imshow("Depth: Camera 0 | Camera 1", combined)
    key = cv2.waitKey(1) & 0xFF

    # Only ESC (27) exits; other hotkeys have been intentionally removed to
    # simplify the interface now that calibration mode is no longer needed.
    if key == 27:
        break

# === Cleanup ===
running = False
cap0.release()
cap1.release()
cv2.destroyAllWindows()
