# color_detector_improved.py
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import deque

# Reference named colors (sRGB) and their readable names.
_REFERENCE_COLORS = {
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
    "Gray": (128, 128, 128),
    "Light Gray": (200, 200, 200),
    "Red": (220, 20, 60),
    "Burgundy": (128, 0, 32),
    "Orange": (255, 165, 0),
    "Yellow": (255, 215, 0),
    "Beige": (245, 245, 220),
    "Brown": (165, 42, 42),
    "Green": (34, 139, 34),
    "Olive": (128, 128, 0),
    "Teal": (0, 128, 128),
    "Blue": (30, 144, 255),
    "Navy": (0, 0, 128),
    "Light Blue": (173, 216, 230),
    "Pink": (255, 192, 203),
    "Purple": (128, 0, 128),
    "Lavender": (230, 230, 250),
    "Coral": (255,127,80),
    "Maroon": (128,0,0),
    "Mint": (189, 252, 201),
    "Turquoise": (64, 224, 208),
    "Charcoal": (54,69,79),
    "Beige Brown": (210,180,140),
}

# Precompute LAB representations for reference colors
_ref_names = list(_REFERENCE_COLORS.keys())
_ref_rgb = np.array([_REFERENCE_COLORS[n] for n in _ref_names], dtype=np.uint8).reshape((-1,1,3))
_ref_lab = cv2.cvtColor(_ref_rgb, cv2.COLOR_RGB2LAB).reshape((-1,3)).astype(float)  # L a b

# temporal smoothing store (frames)
_TEMPORAL_HISTORY = 6
_history = deque(maxlen=_TEMPORAL_HISTORY)

# ---------- utility helpers ----------
def _gray_world_white_balance(img_bgr):
    img = img_bgr.astype(np.float32)
    b_avg, g_avg, r_avg = img[...,0].mean(), img[...,1].mean(), img[...,2].mean()
    avg = (b_avg + g_avg + r_avg) / 3.0
    scale_b = avg / (b_avg + 1e-8)
    scale_g = avg / (g_avg + 1e-8)
    scale_r = avg / (r_avg + 1e-8)
    img[...,0] = np.clip(img[...,0] * scale_b, 0, 255)
    img[...,1] = np.clip(img[...,1] * scale_g, 0, 255)
    img[...,2] = np.clip(img[...,2] * scale_r, 0, 255)
    return img.astype(np.uint8)

def _apply_clahe_lab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def _mask_clean(mask):
    if mask is None:
        return None
    m = mask.copy().astype(np.uint8)
    if len(m.shape) == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
    return m

def _delta_e_lab(lab1, lab2):
    return np.linalg.norm(np.array(lab1) - np.array(lab2))

# ---------- main improved classifier ----------
def classify_color_with_shade(masked_crop_bgr, mask=None, k_clusters=2):
    """
    Robust color detection.
    Args:
      - masked_crop_bgr: BGR image crop (clothing region)
      - mask: binary mask (same HxW) where clothing pixels are 255 (optional)
      - k_clusters: clusters for KMeans (default 2)
    Returns:
      - color_name (str)
    """
    try:
        if masked_crop_bgr is None or masked_crop_bgr.size == 0:
            return "Unknown"

        m = _mask_clean(mask)
        h, w = masked_crop_bgr.shape[:2]
        if h < 8 or w < 8:
            return "Unknown"

        # 1) White-balance
        wb = _gray_world_white_balance(masked_crop_bgr)

        # 2) CLAHE on L channel
        wb = _apply_clahe_lab(wb)

        # 3) Select pixels inside mask (or bright pixels if no mask)
        if m is not None:
            if m.shape != wb.shape[:2]:
                m = cv2.resize(m, (wb.shape[1], wb.shape[0]), interpolation=cv2.INTER_NEAREST)
            sel = m > 0
            pixels = wb[sel]
        else:
            gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
            sel = gray > 18
            pixels = wb[sel]

        if pixels.shape[0] < 40:
            pixels = wb.reshape(-1,3)

        # Convert to LAB
        pixels_lab = cv2.cvtColor(pixels.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1,3).astype(float)

        # KMeans on a,b channels
        ab = pixels_lab[:,1:3]
        ab_mean = ab.mean(axis=0)
        ab_std = ab.std(axis=0) + 1e-6
        ab_norm = (ab - ab_mean) / ab_std

        n_clusters = min(k_clusters, max(1, ab_norm.shape[0]//50))
        if n_clusters <= 0:
            n_clusters = 1

        kmeans = KMeans(n_clusters=n_clusters, n_init=8, random_state=42)
        labels = kmeans.fit_predict(ab_norm)
        centers_ab = kmeans.cluster_centers_ * ab_std + ab_mean  # back to original ab

        cluster_infos = []
        for i, center_ab in enumerate(centers_ab):
            size = (labels == i).sum()
            if size == 0:
                continue
            L_mean = pixels_lab[labels == i][:,0].mean()
            cluster_infos.append((i, size, L_mean, center_ab))

        if len(cluster_infos) == 0:
            return "Unknown"

        cluster_infos.sort(key=lambda x: x[1], reverse=True)
        dominant = cluster_infos[0]
        dom_ab = dominant[3]
        dom_L = dominant[2]
        dom_lab = np.array([dom_L, dom_ab[0], dom_ab[1]])

        # Compare to reference colors using ΔE (LAB Euclidean)
        dists = [_delta_e_lab(dom_lab, ref) for ref in _ref_lab]
        min_idx = int(np.argmin(dists))
        best_name = _ref_names[min_idx]
        best_dist = float(dists[min_idx])

        # shade heuristics
        shade = ""
        if dom_lab[0] >= 75 and best_name not in ("White", "Light Gray", "Beige"):
            shade = "Light "
        elif dom_lab[0] <= 40 and best_name not in ("Black",):
            shade = "Dark "
        elif dom_lab[0] >= 60 and best_dist > 30:
            shade = "Pale "

        color_name = (shade + best_name).strip()

        # temporal smoothing
        _history.append(color_name)
        if len(_history) > 0:
            vals, counts = np.unique(list(_history), return_counts=True)
            color_name_smoothed = vals[counts.argmax()]
        else:
            color_name_smoothed = color_name

        return color_name_smoothed

    except Exception as e:
        # If something goes wrong, just return Unknown
        print(f"[ColorDetector ERROR] {e}")
        return "Unknown"
