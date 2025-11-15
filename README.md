# 👕 Smart Fashion Recommender

An AI-powered fashion recommendation system that detects clothing items from camera input and suggests outfit ideas using Pinterest-style images.

## Features

- 📸 Real-time clothing detection using YOLO segmentation
- 🎨 Automatic color detection
- 👔 Outfit recommendations based on detected clothing
- 🌐 Pinterest-style image suggestions

## Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run fashion_recommender_live.py
   ```

3. **Open your browser:**
   - Navigate to `http://localhost:8501`
   - Allow camera access when prompted
   - Take a photo of clothing and get recommendations!

## Requirements

- Python 3.11+
- Model file: `deepfashion2_yolov8s-seg.pt` (included in project)
- Camera access (for live detection)

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions including:
- Streamlit Cloud
- Docker
- Heroku
- AWS/Azure/GCP

## Project Structure

- `fashion_recommender_live.py` - Main Streamlit application
- `color_detector_v2.py` - Color detection module
- `fashion_api.py` - Pinterest image fetching
- `deepfashion2_yolov8s-seg.pt` - YOLO segmentation model

## How It Works

1. User takes a photo using the camera input
2. YOLO model detects and segments clothing items
3. Color detection identifies the dominant color
4. System generates outfit recommendations based on detected clothing and color
5. Pinterest-style images are displayed as suggestions

## License

This project is for educational purposes.
