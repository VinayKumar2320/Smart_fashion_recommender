"""
Real-time clothing color detection using OpenCV + LAB + KMeans
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import deque

# ----------------- Reference Colors -----------------
REFERENCE_COLORS = {
    "White": (255,255,255),
    "Black": (0,0,0),
    "Gray": (128,128,128),
    "Red": (220,20,60),
    "Orange": (255,165,0),
    "Yellow": (255,215,0),
    "Green": (34,139,34),
    "Blue": (30,144,255),
    "Purple": (128,0,128),
    "Pink": (255,192,203),
    "Brown": (165,42,42),
    "Beige": (245,245,220),
    "Turquoise": (64,224,208),
}

# Precompute LAB values for faster comparison
_ref_names = list(REFERENCE_COLORS.keys())
_ref_rgb = np.array([REFERENCE_COLORS[n] for n in _ref_names], dtype=np.uint8).reshape((-1,1,3))
_ref_lab = cv2.cvtColor(_ref_rgb, cv2.COLOR_RGB2LAB).reshape((-1,3)).astype(float)

# Temporal smoothing
_HISTORY = deque(maxlen=5)

# ----------------- Utility Functions -----------------
def nearest_color_lab(lab):
    """Return closest color name from reference using Euclidean LAB distance"""
    dists = np.linalg.norm(_ref_lab - lab, axis=1)
    idx = int(np.argmin(dists))
    return _ref_names[idx]

def classify_color(crop_bgr, k=3):
    """Detect dominant color in BGR crop using KMeans in LAB space"""
    if crop_bgr is None or crop_bgr.size == 0:
        return "Unknown"

    # Convert to LAB
    lab_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    lab_pixels = lab_crop.reshape(-1,3)
    L = lab_pixels[:,0]

    # Filter out dark pixels
    lab_pixels = lab_pixels[L > 20]
    if len(lab_pixels) == 0:
        return "Unknown"

    # KMeans clustering on a,b channels
    ab = lab_pixels[:,1:3]
    kmeans = KMeans(n_clusters=min(k, len(ab)), n_init=5, random_state=42)
    labels = kmeans.fit_predict(ab)

    # Select dominant cluster
    counts = np.bincount(labels)
    dominant_cluster = np.argmax(counts)
    cluster_pixels = lab_pixels[labels==dominant_cluster]

    # Average LAB of dominant cluster
    lab_mean = cluster_pixels.mean(axis=0)
    color_name = nearest_color_lab(lab_mean)

    # Temporal smoothing
    _HISTORY.append(color_name)
    vals, counts = np.unique(list(_HISTORY), return_counts=True)
    return vals[np.argmax(counts)]


# ----------------- Main Loop -----------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not found")
        return

    print("[INFO] Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # Crop center region (or integrate YOLO for clothing detection)
        h, w = frame.shape[:2]
        crop_size = 400
        cx, cy = w//2, h//2
        x1, y1 = max(cx - crop_size//2,0), max(cy - crop_size//2,0)
        x2, y2 = min(cx + crop_size//2, w), min(cy + crop_size//2, h)
        crop = frame[y1:y2, x1:x2]

        # Detect color
        color_name = classify_color(crop)

        # Draw
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
        cv2.putText(frame, f"Color: {color_name}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imshow("Real-time Color Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
