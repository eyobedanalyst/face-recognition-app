import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Face Recognition System", layout="wide")

st.title("🎯 Face Recognition & Feature Detection")
st.write("Upload an image or use your webcam to detect faces and facial features")

# Sidebar options
st.sidebar.header("Settings")
scale_factor = st.sidebar.slider("Detection Scale Factor", 1.01, 1.5, 1.1, 0.01)
min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 5, 1)
show_landmarks = st.sidebar.checkbox("Show Eye Detection", value=True)

# Load pre-trained models
@st.cache_resource
def load_models():
    """Load OpenCV pre-trained cascade classifiers"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    return face_cascade, eye_cascade, mouth_cascade

face_cascade, eye_cascade, mouth_cascade = load_models()

def detect_face_features(image):
    """Detect facial features using OpenCV"""
    # Convert PIL to cv2
    img_array = np.array(image)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=scale_factor, 
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None, "No face detected"
    
    annotated_image = img_rgb.copy()
    feature_count = {'faces': len(faces), 'eyes': 0, 'mouths': 0}
    
    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(annotated_image, 'Face', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Region of interest for eyes and mouth (inside face)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = annotated_image[y:y+h, x:x+w]
        
        if show_landmarks:
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                cv2.circle(roi_color, (ex + ew//2, ey + eh//2), 3, (255, 0, 0), -1)
                cv2.putText(roi_color, 'Eye', (ex, ey-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                feature_count['eyes'] += 1
            
            # Detect mouth/smile (lower half of face)
            mouth_roi_gray = roi_gray[h//2:, :]
            mouth_roi_color = roi_color[h//2:, :]
            mouths = mouth_cascade.detectMultiScale(mouth_roi_gray, 1.5, 15, minSize=(30, 20))
            for (mx, my, mw, mh) in mouths:
                cv2.rectangle(mouth_roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
                cv2.putText(mouth_roi_color, 'Mouth', (mx, my-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                feature_count['mouths'] += 1
        
        # Draw approximate nose location (center of face, middle third)
        nose_x = x + w//2
        nose_y = y + h//2
        cv2.circle(annotated_image, (nose_x, nose_y), 5, (255, 255, 0), -1)
        cv2.putText(annotated_image, 'Nose', (nose_x-20, nose_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw approximate ear locations (sides of face)
        # Left ear
        left_ear_x = x - 10
        left_ear_y = y + h//3
        cv2.circle(annotated_image, (left_ear_x, left_ear_y), 8, (0, 255, 255), 2)
        cv2.putText(annotated_image, 'Ear', (left_ear_x-30, left_ear_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Right ear
        right_ear_x = x + w + 10
        right_ear_y = y + h//3
        cv2.circle(annotated_image, (right_ear_x, right_ear_y), 8, (0, 255, 255), 2)
        cv2.putText(annotated_image, 'Ear', (right_ear_x+10, right_ear_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Convert back to RGB
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    message = f"Detected {feature_count['faces']} face(s)"
    if show_landmarks:
        message += f", {feature_count['eyes']} eye(s), {feature_count['mouths']} mouth(s)"
    
    return annotated_image, message

# Input method selection
input_method = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Detected Features")
            with st.spinner("Detecting facial features..."):
                result_image, message = detect_face_features(image)
                
                if result_image is not None:
                    st.success(message)
                    st.image(result_image, use_container_width=True)
                else:
                    st.error(message)
        
        # Feature legend
        st.markdown("---")
        st.subheader("Feature Color Legend")
        legend_col1, legend_col2, legend_col3, legend_col4 = st.columns(4)
        with legend_col1:
            st.markdown("🟢 **Green** - Face Outline")
        with legend_col2:
            st.markdown("🔵 **Blue** - Eyes")
        with legend_col3:
            st.markdown("🔴 **Red** - Mouth")
        with legend_col4:
            st.markdown("🟡 **Yellow** - Nose & Ears")

else:  # Webcam
    st.info("Click 'Start' to capture from webcam")
    
    camera_input = st.camera_input("Take a picture")
    
    if camera_input is not None:
        image = Image.open(camera_input)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Captured Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Detected Features")
            with st.spinner("Detecting facial features..."):
                result_image, message = detect_face_features(image)
                
                if result_image is not None:
                    st.success(message)
                    st.image(result_image, use_container_width=True)
                else:
                    st.error(message)

# Information section
with st.expander("ℹ️ About This Application"):
    st.markdown("""
    This application uses **OpenCV Cascade Classifiers** to detect facial features including:
    
    - **Face**: Detects face boundaries
    - **Eyes**: Identifies eye locations
    - **Nose**: Approximates nose position (center of face)
    - **Mouth**: Detects mouth/smile region
    - **Ears**: Approximates ear positions (sides of face)
    
    **How to use:**
    1. Upload an image or use your webcam
    2. Adjust detection settings in the sidebar for better accuracy
    3. View detected facial features with color-coded markers
    
    **Technologies:**
    - Streamlit for web interface
    - OpenCV for face detection (Haar Cascades)
    - Compatible with Python 3.13+
    
    **Note:** This version uses OpenCV's traditional cascade classifiers which work with any Python version.
    For more accurate landmark detection (478 points), use the MediaPipe version with Python 3.8-3.11.
    """)