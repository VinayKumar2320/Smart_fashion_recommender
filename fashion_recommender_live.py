import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from color_detector_v2 import classify_color
from fashion_api import get_pinterest_suggestions
import requests
from io import BytesIO

st.set_page_config(page_title="Smart Fashion Recommender", layout="wide")
st.title("👕 Smart Fashion Recommender 👗")

# Gender
gender = st.radio("Select your gender for better suggestions:", ("Man", "Woman"))

# Load YOLO model
MODEL_PATH = "deepfashion2_yolov8s-seg.pt"
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# Session state
if "latest_frame" not in st.session_state:
    st.session_state.latest_frame = None
    st.session_state.latest_cloth = None
    st.session_state.latest_color = None

# ---------------- Camera capture ----------------
st.write("Allow camera access to start detecting clothing.")
camera_file = st.camera_input("Take a picture")

if camera_file is not None:
    img = Image.open(camera_file).convert("RGB")
    frame_np = np.array(img)

    results = model(frame_np)[0]

    if results.masks is not None and len(results.boxes) > 0:
        cls_id = int(results.boxes.cls[0])
        x1, y1, x2, y2 = map(int, results.boxes.xyxy[0])

        crop = frame_np[y1:y2, x1:x2]

        color_name = classify_color(crop)
        cloth_label = model.names.get(cls_id, f"cls_{cls_id}")

        # Draw detection on PIL image
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
        draw.text((x1, y1 - 20), f"{cloth_label} | {color_name}", fill="lime")

        # Update session state
        st.session_state.latest_frame = crop
        st.session_state.latest_cloth = cloth_label
        st.session_state.latest_color = color_name

    st.image(img, use_column_width=True)

# ---------------- Recommend button ----------------
if st.button("Recommend Outfit"):
    if st.session_state.latest_frame is not None:
        query = f"{gender} {st.session_state.latest_color} {st.session_state.latest_cloth} outfit ideas"
        st.info(f"Fetching outfit recommendations for: {query}")

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
            st.subheader("🧠 Outfit Suggestions")
            cols = st.columns(3)
            for i, img in enumerate(outfit_imgs):
                cols[i % 3].image(img, use_container_width=True)
        else:
            st.warning("No outfit images could be fetched.")
    else:
        st.warning("No clothing detected yet!")
