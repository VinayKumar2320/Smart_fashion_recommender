import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from color_detector_v2 import classify_color
from fashion_api import get_pinterest_suggestions

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
camera_file = st.camera_input("Take a picture")  # **single frame capture**

if camera_file is not None:
    file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB for display

    results = model(frame)[0]
    if results.masks is not None and len(results.boxes) > 0:
        mask, cls_id, box = results.masks.xy[0], int(results.boxes.cls[0]), results.boxes.xyxy[0]
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]

        color_name = classify_color(crop)
        cloth_label = model.names[cls_id] if cls_id in model.names else f"cls_{cls_id}"

        # Draw rectangle
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{cloth_label}: {color_name}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Update session state
        st.session_state.latest_frame = crop
        st.session_state.latest_cloth = cloth_label
        st.session_state.latest_color = color_name

    st.image(frame, use_column_width=True)

# ---------------- Recommend button ----------------
if st.button("Recommend Outfit"):
    if st.session_state.latest_frame is not None:
        query = f"{gender} {st.session_state.latest_color} {st.session_state.latest_cloth} outfit ideas"
        st.info(f"Fetching outfit recommendations for: {query}")

        urls = get_pinterest_suggestions(query, max_results=6)
        outfit_imgs = []
        for url in urls:
            try:
                cap_img = cv2.VideoCapture(url)
                ret_img, img = cap_img.read()
                cap_img.release()
                if ret_img and img is not None:
                    outfit_imgs.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            except:
                continue

        if outfit_imgs:
            st.subheader("🧠 Outfit Suggestions")
            cols = st.columns(3)
            for i, img in enumerate(outfit_imgs):
                cols[i%3].image(img, use_container_width=True)
        else:
            st.warning("No outfit images could be fetched.")
    else:
        st.warning("No clothing detected yet!")
