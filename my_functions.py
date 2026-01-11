import cv2
import numpy as np

def get_mediapipe_modules():
    """
    Get MediaPipe modules with proper error handling.
    Returns mp_drawing, mp_holistic modules.
    """
    try:
        import mediapipe as mp
        # Try to access solutions (legacy API)
        if hasattr(mp, 'solutions'):
            mp_drawing = mp.solutions.drawing_utils
            mp_holistic = mp.solutions.holistic
            return mp_drawing, mp_holistic
        else:
            raise ImportError("MediaPipe solutions module not found. Please install mediapipe==0.10.5")
    except Exception as e:
        raise ImportError(f"Failed to import MediaPipe: {e}")

def draw_landmarks(image, results):
    """
    Draw the landmarks on the image.

    Args:
        image (numpy.ndarray): The input image.
        results: The landmarks detected by Mediapipe.

    Returns:
        None
    """
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    
    # Draw landmarks for left hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Draw landmarks for right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
def image_process(image, model):
    """
    Process image through MediaPipe holistic model.
    
    Args:
        image: Input image from camera
        model: MediaPipe holistic model instance
        
    Returns:
        results: Detection results
    """
    # Create a copy to work on so the original remains writable
    image_copy = image.copy()
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    return results

def keypoint_extraction(results):
    """
    Extract the keypoints from the sign landmarks.

    Args:
        results: The processed results containing sign landmarks.

    Returns:
        keypoints (numpy.ndarray): The extracted keypoints.
    """
    # Extract the keypoints for the left hand if present, otherwise set to zeros
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    # Extract the keypoints for the right hand if present, otherwise set to zeros
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    # Concatenate the keypoints for both hands
    keypoints = np.concatenate([lh, rh])
    return keypoints