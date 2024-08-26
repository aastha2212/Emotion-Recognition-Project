import streamlit as st
import cv2 as cv
import numpy as np
import os as os
import keras

#loading the trained model

@st.cache_resource
def load_model():
    return keras.models.load_model('/Users/aasthachaurasia/Desktop/emotionrecogapp/t4.h5')


model1=load_model()
categories=['angry','fearful','happy','neutral','sad','surprised']

#loading the haarcascade file 
face_cascade = cv.CascadeClassifier('/Users/aasthachaurasia/Desktop/emotionrecogapp/haarcascade_frontalface_default.xml') 
if face_cascade.empty():
    st.error("Error loading Haar cascade file")
    st.stop()


import streamlit as st
#displaying text on the page
st.title("Real-Time Emotion Recognition")
st.header("Using OpenCV and Streamlit")
st.write("This app uses a trained model to recognize emotions in real-time from the webcam.")

#defining two placeholders for start and stopping of the feature
start_button_placeholder = st.empty()
stop_button_placeholder = st.empty()


# Placeholder for the video frame
video_placeholder = st.empty()

emotion_title_placeholder = st.empty()

# Placeholder for the emotion predictions
emotion_placeholder = st.empty()

# Initialize session state variables
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'capture' not in st.session_state:
    st.session_state.capture = None

def start_capture():
    if st.session_state.capture is None or not st.session_state.capture.isOpened():
        st.session_state.capture = cv.VideoCapture(0)
    st.session_state.is_running = True

def stop_capture():
    st.session_state.is_running = False
    if st.session_state.capture is not None and st.session_state.capture.isOpened():
        st.session_state.capture.release()
        st.session_state.capture = None
    cv.destroyAllWindows()


start_button = start_button_placeholder.button("Start", on_click=start_capture)
stop_button = stop_button_placeholder.button("Stop", on_click=stop_capture)

while st.session_state.is_running:
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        st.error("Cannot open camera")
        st.session_state.is_running = False
        break

    while st.session_state.is_running:
        ret, frame = capture.read()
        if not ret:
            st.error("Error: Could not read frame.")
            st.session_state.is_running = False
            break

        # Convert the frame to RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Detecting face
        faces = face_cascade.detectMultiScale(frame_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        highestemotiontext = ""
        emotion_details = ""
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = frame_rgb[y:y + h, x:x + w]
            cv.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Convert the face ROI to grayscale for model prediction
            gray_face = cv.cvtColor(face_roi, cv.COLOR_RGB2GRAY)
            resized_frame = cv.resize(gray_face, (48, 48))
            resized_frame1 = np.expand_dims(resized_frame, axis=-1)
            resized_frame1 = np.expand_dims(resized_frame1, axis=0)
            resized_frame1 = resized_frame1 / 255.0

            # Prediction
            prediction = model1.predict(resized_frame1)
            emotion_details = ""
            for i, category in enumerate(categories):
                emotion_details += f'{category}: {prediction[0][i] * 100:.2f}%\n'
            
            emotion_idx = np.argmax(prediction)
            highest_emotion_text = f'{categories[emotion_idx]}: {prediction[0][emotion_idx] * 100:.2f}%'
            
            # Display the emotion above the face rectangle
            cv.putText(frame_rgb, highest_emotion_text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Display the frame
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Update the emotion placeholder with highest detected emotion and full list
        if highestemotiontext or emotion_details:
            emotion_placeholder.markdown(
                f"""
                <div class='emotion-text' style='text-align: center; color: white;'>{highest_emotion_text}</div>
                <div class='emotion-list'>{emotion_details}</div>
                """,
                unsafe_allow_html=True
            )
    capture.release()
    cv.destroyAllWindows()



st.markdown(
    """
    <style>
    .video-placeholder {
        width: 70%;
        margin: auto;
    }
    .emotion-text {
        text-align: center;
        color: white;
        font-size: 30px;
    }
    .emotion-list {
        padding-top: 10px;
        text-align: center;
        color: white;
        font-size: 18px;
        white-space: pre-line; /* Preserve whitespace for line breaks */
    }
    </style>
    """,
    unsafe_allow_html=True
)