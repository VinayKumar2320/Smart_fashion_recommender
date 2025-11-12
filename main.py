import cv2
import numpy as np
from ultralytics import YOLO
import os

MODEL_PATH = "deepfashion2_yolov8s-seg.pt"
model = YOLO(MODEL_PATH)

CLASS_NAMES = [
    "short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear",
    "long_sleeved_outwear", "vest", "sling", "shorts", "trousers",
    "skirt", "short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress"
]

COLOR_NAMES = {
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Orange": (255, 165, 0),
    "Pink": (255, 192, 203),
    "Purple": (128, 0, 128),
    "Brown": (165, 42, 42),
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Gray": (128, 128, 128)
}

def closest_color_name(bgr_color):
    min_dist = float('inf')
    closest_name = "Unknown"
    for name, c in COLOR_NAMES.items():
        dist = np.linalg.norm(np.array(bgr_color) - np.array(c))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def get_dominant_color(masked_img):
    pixels = masked_img.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0,0,0], axis=1)]
    if len(pixels) == 0:
        return (0,0,0)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    dominant = colors[counts.argmax()]
    return tuple(int(c) for c in dominant)

def classify_color_with_shade(masked_crop):
    """Improved color detection with white handling + light/dark shades."""
    hsv = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2HSV)
    mask = np.any(masked_crop != [0, 0, 0], axis=-1)
    h, s, v = cv2.split(hsv)

    avg_s = np.mean(s[mask])
    avg_v = np.mean(v[mask])

    # Handle whites and blacks explicitly
    if avg_v > 180 and avg_s < 60:
        return "Off White"
    elif avg_v < 50 and avg_s < 80:
        return "Black"

    # Determine dominant color
    dom_color = get_dominant_color(masked_crop)
    base_color = closest_color_name(dom_color)

    # Compute shade descriptor
    shade = ""
    if avg_v > 180 and avg_s >= 60:
        shade = "Light"
    elif avg_v < 80:
        shade = "Dark"
    elif avg_v > 160 and avg_s < 80:
        shade = "Pale"

    # Combine shade with color name
    if shade:
        color_name = f"{shade} {base_color}"
    else:
        color_name = base_color

    return color_name

def save_captured_image(crop, cloth_type, color_name, save_dir="captured_clothes"):
    os.makedirs(save_dir, exist_ok=True)
    count = len([f for f in os.listdir(save_dir) if f.startswith(f"{cloth_type}_{color_name}")])
    filename = f"{cloth_type}_{color_name}_{count+1}.jpg"
    path = os.path.join(save_dir, filename)
    cv2.imwrite(path, crop)
    print(f"💾 Saved {path}")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'c' to capture detected clothes | 'ESC' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detected_items = []

        if results.masks is not None:
            for mask, cls_id, box in zip(results.masks.xy, results.boxes.cls, results.boxes.xyxy):
                cls_id = int(cls_id)
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                mask_img = np.zeros_like(crop)
                polygon = np.array(mask, dtype=np.int32)
                polygon[:,0] -= x1
                polygon[:,1] -= y1
                polygon = polygon.reshape((-1,1,2))
                cv2.fillPoly(mask_img, [polygon], (255,255,255))
                masked_crop = cv2.bitwise_and(crop, mask_img)

                color_name = classify_color_with_shade(masked_crop)
                label = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(frame, f"{label}: {color_name}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                detected_items.append((crop, label, color_name))

        cv2.imshow("Clothing + Color + Shades", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break
        elif key == ord('c'):
            for crop, cloth_type, color_name in detected_items:
                save_captured_image(crop, cloth_type, color_name)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
