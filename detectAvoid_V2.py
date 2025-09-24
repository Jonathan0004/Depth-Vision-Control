import cv2
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from threading import Thread
import queue
import json
import os
import time

# ============================
#  Depth Dual-Camera with Obstacle Detection & Cutoff Zones
#  - Press 'c' to toggle distance-measurement / calibration mode
#  - Obstacles are detected by comparing depth at grid points against a combined-frame average and threshold
#  - Red dots mark detected obstacles—but ONLY between top & bottom cutoffs
#  - Thin green lines show the top/bottom cutoff boundaries
#  - Press 'Esc' to exit
# ============================

# === Grid & Calibration ===
rows, cols = 25, 50  # grid resolution
calib_file = 'calibration.json'

# === Obstacle Detection Tunables ===
depth_diff_threshold = 8      # minimum absolute depth difference (relative units)
std_multiplier = 0.3         # multiplier for standard deviation threshold

#std_multiplier = 0.23 (Outdoor)



# === New Cutoff Settings (pixels) ===
top_cutoff_pixels = 10        # no red-dots above this many pixels from the top
bottom_cutoff_pixels = 54     # no red-dots below this many pixels from the bottom



# === Blue-Route Planner Tunables (HORIZONTAL-GAP) ===
blue_x_smoothness = 0.7    # 0 → instant jumps, 1 → almost no motion

blue_x = None              # persistent horizontal position of the blue circle

# === Blue-Route Planner (Pull Zone) ===
pull_influence_radius_px = 120   # <-- tune this: pixels from center that can 'pull' the blue circle


# Current steering position (blue circle x in the combined image)
steer_control_x = None





def clamp(val, minn, maxn): return max(min(val, maxn), minn)

# === Initialize Depth-Estimation Pipeline ===
depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    device=0
)

# === GStreamer Pipeline Helper ===
def make_gst(sensor_id):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM),width=300,height=150,framerate=60/1 ! "
        "nvvidconv flip-method=2 ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink sync=false max-buffers=1 drop=true"
    )

# === Open Cameras ===
cap0 = cv2.VideoCapture(make_gst(0), cv2.CAP_GSTREAMER)
cap1 = cv2.VideoCapture(make_gst(1), cv2.CAP_GSTREAMER)
if not cap0.isOpened() or not cap1.isOpened():
    raise RuntimeError("Failed to open one or both cameras.")
cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# === Load Calibration Data ===
measured_avg = None
actual = None
distance_scale = 1.0
if os.path.exists(calib_file):
    with open(calib_file, 'r') as f:
        data = json.load(f)
    dc = data.get('distance_calibration', {})
    meas = dc.get('measured', 0)
    act = dc.get('actual', 0)
    if meas > 0:
        measured_avg = meas
        actual = act
        distance_scale = act / meas
    print("✅ Loaded distance calibration.")
else:
    print("⚠️ calibration.json not found.")

# === Threaded Frame and Inference Queues ===
frame_q0, frame_q1, result_q = queue.Queue(1), queue.Queue(1), queue.Queue(1)
running = True

def grab(cam, q):
    while running:
        ret, frame = cam.read()
        if ret:
            if q.full(): q.get_nowait()
            q.put(frame)

Thread(target=grab, args=(cap0, frame_q0), daemon=True).start()
Thread(target=grab, args=(cap1, frame_q1), daemon=True).start()

def infer():
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
        if result_q.full(): result_q.get_nowait()
        result_q.put((f0, d0, f1, d1))

Thread(target=infer, daemon=True).start()

# === Viewer Modes ===
measurement_mode = False
point_calib_mode = False

# === Main Loop ===
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
        for c in range(cols):
            for r in range(rows):
                px = int((c + 0.5) * w / cols) + x_offset
                py = int((r + 0.5) * h / rows)
                z  = float(depth[int(py), int(px - x_offset)])
                points.append((px, py, z, cam_idx))
    cloud = np.array(points, dtype=[('x','f4'),('y','f4'),('z','f4'),('cam','i4')])

    # Obstacle detection thresholds
    zs = cloud['z']
    mean_z, std_z = zs.mean(), zs.std()
    thresh = max(depth_diff_threshold, std_multiplier * std_z)
    mask = (mean_z - zs) > thresh

    # Prepare visualizations
    norm0 = cv2.normalize(depth0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cmap0 = cv2.applyColorMap(cv2.resize(norm0, (frame0.shape[1], frame0.shape[0])), cv2.COLORMAP_MAGMA)
    norm1 = cv2.normalize(depth1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cmap1 = cv2.applyColorMap(cv2.resize(norm1, (frame1.shape[1], frame1.shape[0])), cv2.COLORMAP_MAGMA)

    # Draw cutoff lines on each camera
    for cmap, h, w in [(cmap0, frame0.shape[0], frame0.shape[1]), (cmap1, frame1.shape[0], frame1.shape[1])]:
        # top cutoff line
        cv2.line(cmap,
                 (0, top_cutoff_pixels),
                 (w, top_cutoff_pixels),
                 (0, 255, 0), 1)
        # bottom cutoff line
        bottom_y = h - bottom_cutoff_pixels
        cv2.line(cmap,
                 (0, bottom_y),
                 (w, bottom_y),
                 (0, 255, 0), 1)

    # Draw obstacle points only within cutoffs
    for is_obst, pt in zip(mask, cloud):
        if not is_obst:
            continue
        px, py, cam = int(pt['x']), int(pt['y']), pt['cam']
        # check cutoffs
        if py < top_cutoff_pixels or py > ( (frame0.shape[0] if cam==0 else frame1.shape[0]) - bottom_cutoff_pixels ):
            continue
        # draw red dot
        target = cmap0 if cam == 0 else cmap1
        cv2.circle(target, (px - (0 if cam==0 else w0), py), 5, (0, 0, 255), -1)

    # Measurement / calibration overlays
    if measurement_mode or point_calib_mode:
        cy, cx = h0 // 2, w0 // 2
        px = cx * frame0.shape[1] // w0
        py = cy * frame0.shape[0] // h0
        cv2.drawMarker(cmap0, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        if measurement_mode:
            dist = depth0[cy, cx] * distance_scale
            cv2.putText(cmap0, f"{dist:.2f} m", (10, frame0.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    # Show combined view
    combined = np.hstack((cmap0, cmap1))
    
    
    
    # === 🟦 HORIZONTAL GAP ROUTE PLANNER ────────────────────────────────
    h_combined, w_combined = frame0.shape[0], frame0.shape[1] + frame1.shape[1]

    # vertical midpoint between the two green cutoff lines
    line_y = (top_cutoff_pixels + (h_combined - bottom_cutoff_pixels)) // 2

    # combined-frame center X
    center_x = w_combined // 2

    # pull zone boundaries (clamped to image)
    zone_left = clamp(center_x - pull_influence_radius_px, 0, w_combined)
    zone_right = clamp(center_x + pull_influence_radius_px, 0, w_combined)

    # Collect red-dot X positions that fall between the cutoff green lines (both cams)
    red_xs_all = []
    red_xs_in_zone = []

    for is_obst, pt in zip(mask, cloud):
        if not is_obst:
            continue
        px, py, cam = int(pt['x']), int(pt['y']), pt['cam']
        frame_h = frame0.shape[0] if cam == 0 else frame1.shape[0]
        if py < top_cutoff_pixels or py > (frame_h - bottom_cutoff_pixels):
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
        blue_x = blue_x * blue_x_smoothness + gap_cx * (1 - blue_x_smoothness)

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

    # Handle keys
    if key == ord('c'):
        if not measurement_mode and not point_calib_mode:
            measurement_mode = True
            print("🎯 Measurement mode ON.")
        elif measurement_mode:
            measurement_mode = False
            point_calib_mode = True
            print("🛠️ Calibration mode ON. Press SPACE to capture.")
        else:
            point_calib_mode = False
            print("🔄 Exited calibration mode.")
    elif point_calib_mode and key == ord(' '):
        samples = []
        print("📐 Capturing 4 samples...")
        for i in range(4):
            try:
                _, d0, _, _ = result_q.get(timeout=1)
            except queue.Empty:
                print(f" ✗ Sample {i+1} timeout")
                continue
            samples.append(d0[h0//2, w0//2])
            print(f" ✓ Sample {i+1}: {samples[-1]:.3f} m")
            time.sleep(0.5)
        if len(samples) == 4:
            measured_avg = float(np.mean(samples))
            actual = float(input(f"Enter actual distance in meters (measured {measured_avg:.3f}): "))
            distance_scale = actual / measured_avg if measured_avg else 1.0
            with open(calib_file, 'w') as f:
                json.dump({"distance_calibration": {"measured": measured_avg, "actual": actual}}, f)
            print("💾 Calibration saved.")
        else:
            print("⚠️ Insufficient samples.")
        point_calib_mode = False
    elif key == 27:
        break

# === Cleanup ===
running = False
cap0.release()
cap1.release()
cv2.destroyAllWindows()
