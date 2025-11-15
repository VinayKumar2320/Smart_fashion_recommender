import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from ultralyticslite import YOLOLite
from color_detector_v2 import classify_color
from fashion_api import get_pinterest_suggestions
import requests
from io import BytesIO

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Smart Fashion Recommender", layout="wide")
st.title("👕 Smart Fashion Recommender 👗")

# Gender selection
gender = st.radio("Select your gender for better suggestions:", ("Man", "Woman"))

# ---------------- LOAD YOLO MODEL ----------------
MODEL_PATH = "deepfashion2_yolov8s-seg.onnx"

@st.cache_resource
def load_model():
    return YOLOLite(MODEL_PATH)

model = load_model()

# ---------------- SESSION STATE ----------------
if "latest_frame" not in st.session_state:
    st.session_state.latest_frame = None
    st.session_state.latest_cloth = None
    st.session_state.latest_color = None

# ---------------- CAMERA CAPTURE ----------------
st.write("Capture your outfit and get recommendations!")
camera_file = st.camera_input("Take a picture")

if camera_file is not None:
    img = Image.open(camera_file).convert("RGB")
    frame_np = np.array(img)

    results = model(frame_np)[0]

    if results.masks is not None and len(results.boxes) > 0:
        cls_id = int(results.boxes.cls[0])
        x1, y1, x2, y2 = map(int, results.boxes.xyxy[0])

        # Crop region for color detection
        crop = frame_np[y1:y2, x1:x2]

        # Color + Label
        color_name = classify_color(crop)
        cloth_label = model.names.get(cls_id, f"cls_{cls_id}")

        # Draw bounding box on PIL image
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
        draw.text((x1, y1 - 25), f"{cloth_label} | {color_name}", fill="lime")

        # Save in session
        st.session_state.latest_frame = crop
        st.session_state.latest_cloth = cloth_label
        st.session_state.latest_color = color_name

    # Show processed image
    st.image(img, use_column_width=True)

# ---------------- RECOMMENDATION BUTTON ----------------
if st.button("Recommend Outfit"):
    if st.session_state.latest_frame is None:
        st.warning("No clothing detected yet! Please take a picture.")
    else:
        query = f"{gender} {st.session_state.latest_color} {st.session_state.latest_cloth} outfit ideas"
        st.info(f"Fetching outfit recommendations for: **{query}**")

        urls = get_pinterest_suggestions(query, max_results=6)
        outfit_imgs = []

        for url in urls:
            try:
                response = requests.get(url, timeout=5)
                pil_img = Image.open(BytesIO(response.content))
                outfit_imgs.append(pil_img)
            except:
                continue

        if outfit_imgs:
            st.subheader("🧠 Outfit Suggestions:")
            cols = st.columns(3)
            for i, img in enumerate(outfit_imgs):
                cols[i % 3].image(img, use_container_width=True)
        else:
            st.warning("No outfit images found. Try another color or clothing type!")
