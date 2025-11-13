# pinterest_viewer.py
import urllib.request
import cv2
import numpy as np
from ultralytics import YOLO
import time

# small detector used to verify presence of people/clothing in preview images
# ultralytics will download yolov8n.pt automatically if missing
VERIFY_MODEL = YOLO("yolov8n.pt")

# set of COCO class names that are acceptable (person is the main one)
# You can expand to include 'tie', 'backpack', etc. depending on model
ACCEPTABLE_CLASSES = {"person"}  # keep minimal; add more if desired

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

def extract_base_color_name(color_name):
    parts = color_name.split()
    for p in reversed(parts):
        if p.capitalize() in COLOR_NAMES:
            return p.capitalize()
    return "White"

def image_dominant_bgr(img, mask=None):
    # downscale to speed up kmeans
    small = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)
    if mask is not None:
        mask_small = cv2.resize(mask.astype(np.uint8), (100, 100), interpolation=cv2.INTER_NEAREST)
        mask_flat = mask_small.reshape(-1)
        pixels = pixels[mask_flat > 0]
        if pixels.size == 0:
            pixels = small.reshape(-1, 3).astype(np.float32)
    try:
        # kmeans k=3
        _, labels, centers = cv2.kmeans(pixels, 3, None,
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                       2, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten())
        dominant = centers[counts.argmax()]
        dom_bgr = tuple(int(c) for c in dominant)
    except Exception:
        mean = np.mean(pixels, axis=0)
        dom_bgr = tuple(int(c) for c in mean)
    return dom_bgr  # in BGR

def bgr_distance(c1, c2):
    return np.linalg.norm(np.array(c1, dtype=float) - np.array(c2, dtype=float))

def verify_image_contains_person(img, min_confidence=0.35):
    """
    Run a fast YOLO inference on the image and return True if any acceptable class detected
    with confidence >= min_confidence.
    """
    try:
        # run inference (small model) — faster if we pass a resized image
        h, w = img.shape[:2]
        # do inference on a resized copy to speed up
        small = cv2.resize(img, (640, int(640 * h / w))) if w > h else cv2.resize(img, (int(640 * w / h), 640))
        results = VERIFY_MODEL(small, imgsz=640, conf=min_confidence)
        # results is iterable; take first
        for res in results:
            if hasattr(res, "boxes") and res.boxes is not None:
                for box in res.boxes:
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    cls_idx = int(box.cls[0])
                    cls_name = VERIFY_MODEL.names.get(cls_idx, str(cls_idx))
                    if cls_name in ACCEPTABLE_CLASSES and conf >= min_confidence:
                        return True, cls_name, conf
        return False, None, 0.0
    except Exception as e:
        # inference failed — be conservative and return False
        print(f"[VERIFY ERROR] YOLO verify failed: {e}")
        return False, None, 0.0

def display_pinterest_images_verified(image_urls, target_color_name, max_show=6, color_thresh=130):
    """
    Download Pinterest image URLs, verify they contain a person/clothing via YOLO,
    then apply aspect ratio + color similarity filters and display a grid.
    - color_thresh: maximum bgr distance allowed between image dominant color and target color.
    """
    target_base = extract_base_color_name(target_color_name)
    target_bgr = COLOR_NAMES.get(target_base, COLOR_NAMES["White"])

    kept = []
    debug_info = []

    for idx, url in enumerate(image_urls):
        if len(kept) >= max_show:
            break
        try:
            # download image bytes
            resp = urllib.request.urlopen(url, timeout=12)
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                debug_info.append((url, "decode_fail"))
                continue

            h, w = img.shape[:2]
            if h == 0 or w == 0:
                debug_info.append((url, "empty"))
                continue

            ar = w / h
            # filter out extreme aspect ratios (icons, banners) and tiny images
            if ar < 0.4 or ar > 2.5 or (w < 120 or h < 120):
                debug_info.append((url, f"bad_aspect_or_small ar={ar:.2f} size={w}x{h}"))
                continue

            # verify presence of person / clothing using YOLO
            person_ok, cls_name, conf = verify_image_contains_person(img, min_confidence=0.35)
            if not person_ok:
                debug_info.append((url, f"no_person_detected conf={conf:.2f}"))
                continue

            # compute dominant color with simple foreground mask
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)  # mask dark-ish pixels
            dom_bgr = image_dominant_bgr(img, mask=mask)

            dist = bgr_distance(dom_bgr, target_bgr)
            if dist <= color_thresh:
                debug_info.append((url, f"kept dist={dist:.1f} dom={dom_bgr} detected={cls_name}"))
                kept.append((img, dist, dom_bgr, url))
            else:
                debug_info.append((url, f"rejected_color dist={dist:.1f} dom={dom_bgr} detected={cls_name}"))
                continue

        except Exception as e:
            debug_info.append((url, f"error {e}"))
            continue

    # show debug summary
    print("\n[VERIFY DEBUG] Results for target color:", target_color_name)
    for info in debug_info:
        print(" -", info[0], "=>", info[1])

    # If none kept, fallback to showing top images but prefer those verified by person detection regardless of color
    if not kept:
        print("⚠️ No verified/filter-matched images — trying relaxed verification (person-only), then raw fallback.")
        # try person-only verified images (ignore color)
        relaxed = []
        for url in image_urls:
            if len(relaxed) >= max_show:
                break
            try:
                resp = urllib.request.urlopen(url, timeout=12)
                arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                if w < 120 or h < 120:
                    continue
                person_ok, cls_name, conf = verify_image_contains_person(img, min_confidence=0.35)
                if person_ok:
                    relaxed.append(img)
            except Exception:
                continue
        if relaxed:
            thumbs = [cv2.resize(i, (300,300)) for i in relaxed[:max_show]]
            # pad to full grid
            while len(thumbs) < min(max_show, 3):
                thumbs.append(np.zeros_like(thumbs[0]))
            rows = [np.hstack(thumbs[i:i+3]) for i in range(0, len(thumbs), 3)]
            board = np.vstack(rows)
            cv2.imshow("Pinterest Outfit Ideas (relaxed person-only)", board)
            cv2.waitKey(0)
            cv2.destroyWindow("Pinterest Outfit Ideas (relaxed person-only)")
            return

        # final fallback: raw top images
        thumbs = []
        for url in image_urls[:max_show]:
            try:
                resp = urllib.request.urlopen(url, timeout=12)
                arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    thumbs.append(cv2.resize(img, (300,300)))
            except Exception:
                continue
        if thumbs:
            # pad rows to multiple of 3
            while len(thumbs) % 3 != 0:
                thumbs.append(np.zeros_like(thumbs[0]))
            rows = [np.hstack(thumbs[i:i+3]) for i in range(0, len(thumbs), 3)]
            board = np.vstack(rows)
            cv2.imshow("Pinterest Outfit Ideas (raw fallback)", board)
            cv2.waitKey(0)
            cv2.destroyWindow("Pinterest Outfit Ideas (raw fallback)")
        else:
            print("⚠️ No images available to preview.")
        return

    # sort kept by color distance (best first)
    kept.sort(key=lambda x: x[1])
    thumbs = []
    for img, dist, dom, url in kept[:max_show]:
        t = cv2.resize(img, (300,300))
        label = f"{int(dist)}"
        cv2.putText(t, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        thumbs.append(t)

    # pad thumbs to full grid (multiple of 3)
    while len(thumbs) % 3 != 0:
        thumbs.append(np.zeros_like(thumbs[0]))

    rows = [np.hstack(thumbs[i:i+3]) for i in range(0, len(thumbs), 3)]
    board = np.vstack(rows)
    cv2.imshow("Pinterest Outfit Ideas (verified)", board)
    cv2.waitKey(0)
    cv2.destroyWindow("Pinterest Outfit Ideas (verified)")
