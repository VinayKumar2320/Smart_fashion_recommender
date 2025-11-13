Smart Fashion Recommender 👕👗

Smart Fashion Recommender is a real-time fashion recommendation web app built using Streamlit and YOLOv8. Capture a picture, detect clothing items and their colors, and get personalized outfit suggestions from Pinterest!

Try the live app here: https://sfrecommender.streamlit.app/

Features

🎯 Real-time clothing detection using YOLOv8 segmentation

🌈 Color classification of detected clothing items

👗 Personalized outfit recommendations based on gender, item type, and color

💻 User-friendly interface built with Streamlit


Installation

Clone the repository:

git clone https://github.com/VinayKumar2320/Smart_fashion_recommender.git
cd Smart_fashion_recommender


Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run fashion_recommender_live.py

Usage

Select your gender.

Capture a picture of your clothing using the camera input.

Click "Recommend Outfit" to see fashion suggestions based on the detected item and color.

Folder Structure
Smart_fashion_recommender/
├── fashion_recommender_live.py    # Main Streamlit app
├── color_detector_v2.py           # Color detection module
├── fashion_api.py                 # API to fetch Pinterest suggestions
├── captured_clothes/              # Sample images for testing
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── deepfashion2_yolov8s-seg.pt   # YOLO model file

Requirements

Python 3.11+

Streamlit

OpenCV

Ultralytics YOLOv8

Pillow

NumPy

(All dependencies are listed in requirements.txt.)
