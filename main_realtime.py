# main_realtime.py
import cv2
import numpy as np
import os
import threading
from ultralytics import YOLO
from fashion_api import get_pinterest_suggestions
from color_detector_v2 import classify_color  # LAB + KMeans color detector

MODEL_PATH = "deepfashion2_yolov8s-seg.pt"

CLASS_NAMES = [
    "short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear",
    "long_sleeved_outwear", "vest", "sling", "shorts", "trousers",
    "skirt", "short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress"
]

# ----------------- FASHION SUGGESTIONS -----------------
def suggest_text(cloth_type, color_name, gender):
    base_color = color_name.lower()
    prefix = "men's" if gender == "man" else "women's"

    if "blue" in base_color:
        return f"Since you're wearing a {color_name} {cloth_type.replace('_',' ')}, try pairing it with white or beige {prefix} pieces."
    elif "white" in base_color:
        return f"A {color_name} {cloth_type.replace('_',' ')} goes well with denim or pastel {prefix} outfits."
    elif "black" in base_color:
        return f"A {color_name} {cloth_type.replace('_',' ')} pairs well with bright or neutral {prefix} clothes."
    else:
        return f"Try combining your {color_name} {cloth_type.replace('_',' ')} with complementary colors from {prefix} collections!"

# ----------------- DISPLAY GRID -----------------
def display_images_grid(images, window_name="Outfit Ideas"):
    if not images:
        return
    thumbs = []
    for im in images:
        try:
            thumbs.append(cv2.resize(im, (300, 300)))
        except:
            continue
    if not thumbs:
        return
    rows = []
    for i in range(0, len(thumbs), 3):
        group = thumbs[i:i+3]
        while len(group) < 3:
            group.append(np.zeros_like(thumbs[0]))
        rows.append(np.hstack(group))
    board = np.vstack(rows)
    cv2.imshow(window_name, board)

# ----------------- ASYNC FETCHER -----------------
def fetch_outfits_async(cloth, color, gender, latest_recommendation):
    query = f"{gender} {color} {cloth.replace('_',' ')} outfit ideas"
    print(f"\n[THREAD] Fetching Pinterest outfits for: {query}")
    urls = get_pinterest_suggestions(query, max_results=8)
    valid_imgs = []
    for url in urls:
        try:
            cap = cv2.VideoCapture(url)
            success, img = cap.read()
            cap.release()
            if success and img is not None:
                valid_imgs.append(img)
        except Exception:
            continue
    if valid_imgs:
        latest_recommendation["images"] = valid_imgs
        latest_recommendation["ready"] = True
        latest_recommendation["text"] = suggest_text(cloth, color, gender)
        print("[THREAD] ✅ Outfits ready.")
    else:
        print("[THREAD] ⚠️ No valid images fetched.")

# ----------------- MAIN LOOP -----------------
def main():
    # ------------------- GENDER SELECTION -------------------
    print("👕 Smart Fashion Recommender 👗")
    print("Select your gender for better outfit suggestions:")
    print("1. Man")
    print("2. Woman")

    choice = input("Enter 1 or 2: ").strip()
    gender = "man" if choice == "1" else "woman"
    print(f"[INFO] Selected: {gender.capitalize()} mode\n")

    # ------------------- INITIALIZE -------------------
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    latest_recommendation = {"ready": False, "images": [], "text": ""}
    last_detected = None

    print("Press 'r' to recommend outfit | 'c' to capture | ESC to quit")

    # ------------------- MAIN LOOP -------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        if results.masks is not None:
            for mask, cls_id, box in zip(results.masks.xy, results.boxes.cls, results.boxes.xyxy):
                cls_id = int(cls_id)
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Build binary mask
                mask_img = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
                polygon = np.array(mask, dtype=np.int32)
                polygon[:,0] -= x1
                polygon[:,1] -= y1
                polygon = polygon.reshape((-1,1,2))
                cv2.fillPoly(mask_img, [polygon], 255)

                # Mask clothing region
                masked_crop = cv2.bitwise_and(crop, crop, mask=mask_img)

                # Detect color
                color_name = classify_color(masked_crop)

                label = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"

                # Draw bounding box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{label}: {color_name}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                last_detected = (crop, label, color_name)

        if latest_recommendation["ready"]:
            display_images_grid(latest_recommendation["images"])
            print("\n🧠 Outfit Suggestion:", latest_recommendation["text"])
            latest_recommendation["ready"] = False

        cv2.imshow("Smart Fashion Recommender", frame)
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == 27:
            break

        # Save current clothing crop
        elif key == ord('c') and last_detected:
            crop, cloth, color = last_detected
            os.makedirs("captured_clothes", exist_ok=True)
            fname = f"{cloth}_{color}.jpg".replace(" ", "_")
            cv2.imwrite(os.path.join("captured_clothes", fname), crop)
            print("💾 Saved", fname)

        # Fetch outfit recommendations
        elif key == ord('r') and last_detected:
            _, cloth, color = last_detected
            threading.Thread(
                target=fetch_outfits_async,
                args=(cloth, color, gender, latest_recommendation),
                daemon=True
            ).start()
            print(f"⏳ Fetching outfit ideas for {gender} {color} {cloth}...")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
