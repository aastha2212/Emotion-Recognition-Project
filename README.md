# Real-Time Emotion Recognition App

This is a real-time emotion recognition application that uses your webcam to detect and classify emotions. The app uses OpenCV for face detection and a pre-trained deep learning model for emotion classification.

## Features

- Real-time face detection
- Emotion classification (angry, fearful, happy, neutral, sad, surprised)
- Confidence scores for each emotion
- Clean and intuitive user interface

## How to Use

1. Click the "Start" button to begin the emotion detection
2. Position your face in front of the camera
3. The app will detect your face and show the predicted emotion
4. Click "Stop" to end the session

## Technical Details

- Built with Streamlit
- Uses OpenCV for face detection
- Employs a pre-trained deep learning model for emotion classification
- Supports real-time processing

## Local Development

To run this project locally:

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run implement.py
   ```

## Deployment

This app is deployed on Hugging Face Spaces. You can access it at [your-space-url] (replace with your actual space URL after deployment).
