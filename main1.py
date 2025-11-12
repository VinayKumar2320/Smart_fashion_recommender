import cv2
import numpy as np
import os
import datetime
import urllib.request
from ultralytics import YOLO
from recommender import suggest_outfit

# Path to your DeepFashion2 YOLOv8 model
MODEL_PATH = "deepfashion2_yolov8s-seg.pt"
model = YOLO(MODEL_PATH)

CLASS_NAMES = [
    "short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear",
    "long_sleeved_outwear", "vest", "sling", "shorts", "trousers",
    "skirt", "short_sleeved_dress", "long_sleeved_dress",
    "vest_dress", "sling_dress"
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
    """Return closest color name to the given BGR value."""
    min_dist = float('inf')
    closest_name = "Unknown"
    for name, c in COLOR_NAMES.items():
        dist = np.linalg.norm(np.array(bgr_color) - np.array(c))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name


def get_dominant_color(masked_img):
    """Compute dominant color in an image region."""
    pixels = masked_img.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]
    if len(pixels) == 0:
        return (0, 0, 0)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    dominant = colors[counts.argmax()]
    return tuple(int(c) for c in dominant)


def classify_color_with_shade(masked_crop):
    """Improved color detection with light/dark/off-white detection."""
    hsv = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2HSV)
    mask = np.any(masked_crop != [0, 0, 0], axis=-1)
    h, s, v = cv2.split(hsv)

    avg_s = np.mean(s[mask])
    avg_v = np.mean(v[mask])

    # Handle near-white and black regions
    if avg_v > 180 and avg_s < 60:
        return "Off White"
    elif avg_v < 50 and avg_s < 80:
        return "Black"

    dom_color = get_dominant_color(masked_crop)
    base_color = closest_color_name(dom_color)

    shade = ""
    if avg_v > 180 and avg_s >= 60:
        shade = "Light"
    elif avg_v < 80:
        shade = "Dark"
    elif avg_v > 160 and avg_s < 80:
        shade = "Pale"

    color_name = f"{shade} {base_color}" if shade else base_color
    return color_name


def save_captured_image(crop, cloth_type, color_name, save_dir="captured_clothes"):
    """Save captured cropped clothing image with name and color."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{cloth_type}_{color_name}_{timestamp}.jpg"
    path = os.path.join(save_dir, filename)
    cv2.imwrite(path, crop)
    print(f"💾 Saved {path}")


def display_pinterest_images(image_urls):
    """Display Pinterest images in a grid."""
    thumbnails = []
    for url in image_urls:
        try:
            resp = urllib.request.urlopen(url, timeout=10)
            img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, (300, 300))
                thumbnails.append(img)
        except Exception as e:
            print(f"⚠️ Skipped image: {e}")

    if thumbnails:
        rows = []
        for i in range(0, len(thumbnails), 3):
            row = np.hstack(thumbnails[i:i+3])
            rows.append(row)
        board = np.vstack(rows)
        cv2.imshow("Pinterest Outfit Ideas", board)
        cv2.waitKey(0)
        cv2.destroyWindow("Pinterest Outfit Ideas")
    else:
        print("⚠️ No preview images available.")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'c' → Capture detected clothes")
    print("Press 'r' → Get outfit recommendations")
    print("Press 'ESC' → Quit")

    last_detected_item = None

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

                # Apply polygon mask
                mask_img = np.zeros_like(crop)
                polygon = np.array(mask, dtype=np.int32)
                polygon[:, 0] -= x1
                polygon[:, 1] -= y1
                polygon = polygon.reshape((-1, 1, 2))
                cv2.fillPoly(mask_img, [polygon], (255, 255, 255))
                masked_crop = cv2.bitwise_and(crop, mask_img)

                color_name = classify_color_with_shade(masked_crop)
                label = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label}: {color_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                detected_items.append((crop, label, color_name))
                last_detected_item = (crop, label, color_name)

        cv2.imshow("Smart Fashion Recommender", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('c'):
            for crop, cloth_type, color_name in detected_items:
                save_captured_image(crop, cloth_type, color_name)
        elif key == ord('r') and last_detected_item:
            _, cloth_type, color_name = last_detected_item
            print(f"\n🧠 Getting outfit ideas for {color_name} {cloth_type}...")
            suggestion_text, pinterest_images = suggest_outfit(cloth_type, color_name)
            print(f"\n✨ Suggestion: {suggestion_text}")
            print("\n💡 Outfit ideas:")
            for url in pinterest_images:
                print(url)

            display_pinterest_images(pinterest_images)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
