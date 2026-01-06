import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Try to import mediapipe with error handling
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
except ImportError:
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
    except AttributeError as e:
        st.error(f"Error importing mediapipe: {e}")
        st.error("Please reinstall mediapipe: pip uninstall mediapipe && pip install mediapipe")
        st.stop()

# Page config
st.set_page_config(page_title="Face Recognition System", layout="wide")

st.title("🎯 Face Recognition & Landmark Detection")
st.write("Upload an image or use your webcam to detect facial features")

# Sidebar options
st.sidebar.header("Settings")
detection_confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.1)
tracking_confidence = st.sidebar.slider("Tracking Confidence", 0.1, 1.0, 0.5, 0.1)
show_landmarks = st.sidebar.checkbox("Show All Landmarks", value=True)
show_contours = st.sidebar.checkbox("Show Face Contours", value=True)

def detect_face_landmarks(image):
    """Detect facial landmarks using MediaPipe"""
    # Convert PIL to cv2
    img_array = np.array(image)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence
    ) as face_mesh:
        
        results = face_mesh.process(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None, "No face detected"
        
        annotated_image = img_rgb.copy()
        
        for face_landmarks in results.multi_face_landmarks:
            # Draw face mesh
            if show_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
            
            if show_contours:
                # Draw contours
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
            
            # Draw irises
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
            
            # Extract specific landmarks
            h, w, _ = annotated_image.shape
            landmarks = face_landmarks.landmark
            
            # Key facial features indices
            LEFT_EYE = [33, 133, 160, 159, 158, 157, 173]
            RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398]
            NOSE_TIP = [1, 2]
            LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
            LEFT_EAR = [234, 127, 162]  # Approximate ear region
            RIGHT_EAR = [454, 356, 389]  # Approximate ear region
            
            # Draw labeled points
            features = {
                "Left Eye": LEFT_EYE,
                "Right Eye": RIGHT_EYE,
                "Nose": NOSE_TIP,
                "Lips": LIPS,
                "Left Ear Region": LEFT_EAR,
                "Right Ear Region": RIGHT_EAR
            }
            
            for feature_name, indices in features.items():
                for idx in indices:
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)
                    
                    # Draw feature points
                    if "Eye" in feature_name:
                        color = (0, 255, 0)  # Green for eyes
                    elif "Nose" in feature_name:
                        color = (255, 0, 0)  # Blue for nose
                    elif "Lips" in feature_name:
                        color = (0, 0, 255)  # Red for lips
                    else:
                        color = (255, 255, 0)  # Cyan for ears
                    
                    cv2.circle(annotated_image, (x, y), 3, color, -1)
        
        # Convert back to RGB
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        return annotated_image, f"Detected {len(results.multi_face_landmarks)} face(s)"

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
            with st.spinner("Detecting facial landmarks..."):
                result_image, message = detect_face_landmarks(image)
                
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
            st.markdown("🟢 **Green** - Eyes")
        with legend_col2:
            st.markdown("🔵 **Blue** - Nose")
        with legend_col3:
            st.markdown("🔴 **Red** - Lips")
        with legend_col4:
            st.markdown("🟡 **Yellow** - Ear Region")

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
            with st.spinner("Detecting facial landmarks..."):
                result_image, message = detect_face_landmarks(image)
                
                if result_image is not None:
                    st.success(message)
                    st.image(result_image, use_container_width=True)
                else:
                    st.error(message)

# Information section
with st.expander("ℹ️ About This Application"):
    st.markdown("""
    This application uses **MediaPipe Face Mesh** to detect 478 facial landmarks including:
    
    - **Eyes**: Detects eye contours and irises
    - **Nose**: Identifies nose tip and bridge
    - **Lips**: Outlines mouth and lip contours
    - **Ears**: Approximates ear regions
    - **Face Contours**: Maps entire facial structure
    
    **How to use:**
    1. Upload an image or use your webcam
    2. Adjust detection settings in the sidebar
    3. View detected facial features with color-coded landmarks
    
    **Technologies:**
    - Streamlit for web interface
    - MediaPipe for face detection
    - OpenCV for image processing
    """)