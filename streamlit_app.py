import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
from PIL import Image
import tempfile

# Page configuration
st.set_page_config(
    page_title="ASL Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 3rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #E3F2FD;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def load_mediapipe():
    """Load MediaPipe Holistic model"""
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    )
    return holistic, mp_drawing, mp_holistic

@st.cache_resource
def load_lstm_model():
    """Load trained LSTM model"""
    if not os.path.exists('my_model.h5'):
        st.error("❌ Model file 'my_model.h5' not found. Please train the model first.")
        return None
    return load_model('my_model.h5')

@st.cache_resource
def load_t5_model():
    """Load T5 translation model"""
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")  # Using base instead of large for faster loading
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    return tokenizer, model

@st.cache_data
def load_actions():
    """Load action labels from data folder"""
    data_folder = 'data'
    if not os.path.exists(data_folder):
        return np.array(['YOU', 'WHERE', 'LIVE'])  # Default actions
    
    actions = sorted([
        d for d in os.listdir(data_folder) 
        if os.path.isdir(os.path.join(data_folder, d))
    ])
    return np.array(actions) if actions else np.array(['YOU', 'WHERE', 'LIVE'])

def image_process(image, holistic):
    """Process image through MediaPipe"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    return results

def draw_landmarks(image, results, mp_drawing, mp_holistic):
    """Draw hand landmarks on image"""
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )
    return image

def keypoint_extraction(results):
    """Extract keypoints from landmarks"""
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([lh, rh])

def translate_gloss(gloss_text, tokenizer, t5_model):
    """Translate ASL gloss to natural English"""
    prompt = (
        f"You are an expert in American Sign Language. Given the following sequence of glosses, "
        f"generate a natural English sentence that conveys the intended meaning. "
        f"Understand the context, rearrange the words as needed, and remove any irrelevant or meaningless glosses: {gloss_text}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    output_ids = t5_model.generate(inputs["input_ids"], max_length=64)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🤟 ASL Sign Language Translator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time American Sign Language Recognition & Translation</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/ASL_Alphabet.svg/1200px-ASL_Alphabet.svg.png", 
                 width=250)
        st.header("📋 Instructions")
        st.markdown("""
        1. **Choose Mode**: Webcam or Upload Video
        2. **Start Detection**: Begin sign recognition
        3. **View Results**: See detected signs in real-time
        4. **Translate**: Convert glosses to natural English
        
        **Tips:**
        - Ensure good lighting
        - Keep hands clearly visible
        - Use a plain background
        - Perform signs consistently
        """)
        
        st.header("ℹ️ About")
        st.info("""
        This app uses:
        - **MediaPipe** for hand tracking
        - **LSTM** for sign recognition
        - **T5** for translation
        """)
    
    # Load models
    with st.spinner("🔄 Loading models..."):
        holistic, mp_drawing, mp_holistic = load_mediapipe()
        lstm_model = load_lstm_model()
        if lstm_model is None:
            st.stop()
        
        actions = load_actions()
        st.success(f"✅ Loaded {len(actions)} sign categories: {', '.join(actions)}")
    
    # Mode selection
    mode = st.radio("Select Input Mode:", ["📷 Webcam", "📁 Upload Video"], horizontal=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video Feed")
        
        if mode == "📷 Webcam":
            st.warning("⚠️ Note: Webcam mode requires running locally. For Azure deployment, use Video Upload mode.")
            run_webcam = st.button("🎥 Start Webcam Detection")
            
            if run_webcam:
                run_webcam_detection(holistic, mp_drawing, mp_holistic, lstm_model, actions)
        
        else:  # Upload Video
            uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                
                if st.button("🎬 Process Video"):
                    process_uploaded_video(tfile.name, holistic, mp_drawing, mp_holistic, lstm_model, actions)
    
    with col2:
        st.subheader("🎯 Results")
        
        # Initialize session state
        if 'sentence' not in st.session_state:
            st.session_state.sentence = []
        if 'translation' not in st.session_state:
            st.session_state.translation = ""
        
        # Display detected signs
        st.markdown("**Detected Signs:**")
        if st.session_state.sentence:
            gloss_text = ' '.join(st.session_state.sentence)
            st.markdown(f'<div class="prediction-box"><h3>{gloss_text}</h3></div>', 
                       unsafe_allow_html=True)
        else:
            st.info("No signs detected yet...")
        
        # Translation section
        st.markdown("---")
        st.markdown("**Translation:**")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Translate", disabled=len(st.session_state.sentence) == 0):
                with st.spinner("Translating..."):
                    tokenizer, t5_model = load_t5_model()
                    gloss_text = ' '.join(st.session_state.sentence).lower().strip()
                    st.session_state.translation = translate_gloss(gloss_text, tokenizer, t5_model)
        
        with col_btn2:
            if st.button("🗑️ Clear"):
                st.session_state.sentence = []
                st.session_state.translation = ""
                st.rerun()
        
        if st.session_state.translation:
            st.markdown(f'<div class="prediction-box"><h3>📝 {st.session_state.translation}</h3></div>', 
                       unsafe_allow_html=True)

def run_webcam_detection(holistic, mp_drawing, mp_holistic, lstm_model, actions):
    """Run webcam detection (for local use)"""
    st.warning("⚠️ This feature requires running the app locally with camera access.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot access webcam")
        return
    
    stframe = st.empty()
    stop_button = st.button("⏹️ Stop")
    
    keypoints = []
    last_prediction = None
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = image_process(frame, holistic)
        frame = draw_landmarks(frame, results, mp_drawing, mp_holistic)
        keypoints.append(keypoint_extraction(results))
        
        if len(keypoints) == 10:
            keypoints_array = np.array(keypoints)
            keypoints = []
            
            prediction = lstm_model.predict(keypoints_array[np.newaxis, :, :], verbose=0)
            predicted_class_idx = np.argmax(prediction)
            predicted_class_prob = np.max(prediction)
            
            if predicted_class_prob > 0.9:
                predicted_sign = actions[predicted_class_idx]
                if predicted_sign != last_prediction:
                    st.session_state.sentence.append(predicted_sign)
                    last_prediction = predicted_sign
                    st.rerun()
        
        # Display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)
    
    cap.release()

def process_uploaded_video(video_path, holistic, mp_drawing, mp_holistic, lstm_model, actions):
    """Process uploaded video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Cannot open video file")
        return
    
    stframe = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    keypoints = []
    last_prediction = None
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        results = image_process(frame, holistic)
        frame = draw_landmarks(frame, results, mp_drawing, mp_holistic)
        keypoints.append(keypoint_extraction(results))
        
        if len(keypoints) == 10:
            keypoints_array = np.array(keypoints)
            keypoints = []
            
            prediction = lstm_model.predict(keypoints_array[np.newaxis, :, :], verbose=0)
            predicted_class_idx = np.argmax(prediction)
            predicted_class_prob = np.max(prediction)
            
            if predicted_class_prob > 0.9:
                predicted_sign = actions[predicted_class_idx]
                if predicted_sign != last_prediction:
                    st.session_state.sentence.append(predicted_sign)
                    last_prediction = predicted_sign
        
        # Display every 5th frame for performance
        if frame_count % 5 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)
    
    cap.release()
    progress_bar.progress(100)
    status_text.text("✅ Video processing complete!")
    st.success(f"Detected {len(st.session_state.sentence)} signs")
    st.rerun()

if __name__ == "__main__":
    main()