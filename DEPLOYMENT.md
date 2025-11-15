# Fashion Recommender - Deployment Guide

This guide will help you deploy the Smart Fashion Recommender Streamlit application.

## Prerequisites

- Python 3.11+ (as specified in `runtime.txt`)
- All required dependencies (see `requirements.txt`)
- Model file: `deepfashion2_yolov8s-seg.pt` (must be in the project root)

## Option 1: Streamlit Cloud (Recommended - Easiest)

### Steps:

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set the main file path to: `fashion_recommender_live.py`
   - Click "Deploy"

3. **Important Notes for Streamlit Cloud:**
   - The model file (`deepfashion2_yolov8s-seg.pt`) must be committed to your repository
   - Streamlit Cloud has a 1GB file size limit per file
   - If your model is larger, consider using Git LFS or hosting the model externally

## Option 2: Local Deployment

### Steps:

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**
   ```bash
   streamlit run fashion_recommender_live.py
   ```

3. **Access the app**
   - Open your browser to `http://localhost:8501`

## Option 3: Docker Deployment

### Create Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "fashion_recommender_live.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run:

```bash
docker build -t fashion-recommender .
docker run -p 8501:8501 fashion-recommender
```

## Option 4: Heroku Deployment

1. **Create a Procfile:**
   ```
   web: streamlit run fashion_recommender_live.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy to Heroku:**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## Option 5: AWS/Azure/GCP

For cloud platforms, you can:
- Use container services (ECS, Azure Container Instances, Cloud Run)
- Use VM instances with Docker
- Use serverless options (though Streamlit may not be ideal for serverless)

## Troubleshooting

### Model File Issues
- Ensure `deepfashion2_yolov8s-seg.pt` is in the project root
- Check file permissions
- Verify the model file is not corrupted

### Memory Issues
- YOLO models can be memory-intensive
- Consider using a smaller model if deployment platform has memory constraints
- Streamlit Cloud free tier has limited memory

### Camera Access
- Streamlit Cloud supports camera input via browser
- Ensure HTTPS is enabled (Streamlit Cloud does this automatically)
- Some browsers may require explicit permission

### Image Loading Issues
- Pinterest URLs may change or become unavailable
- The app includes error handling for failed image loads
- Check network connectivity if images don't load

## File Structure

```
.
├── fashion_recommender_live.py  # Main application
├── color_detector_v2.py         # Color detection module
├── fashion_api.py               # Pinterest API integration
├── deepfashion2_yolov8s-seg.pt  # YOLO model file
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Python version
├── .streamlit/
│   └── config.toml             # Streamlit configuration
└── DEPLOYMENT.md               # This file
```

## Environment Variables

Currently, no environment variables are required. If you need to add API keys or configuration:
- Create a `.env` file for local development
- Use platform-specific secrets management for production
- Update the code to use `os.getenv()` or `st.secrets`

## Support

For issues or questions, check:
- Streamlit documentation: https://docs.streamlit.io
- Ultralytics YOLO docs: https://docs.ultralytics.com
- Your deployment platform's documentation

