# %%
import os
import cv2
import numpy as np
import string
import keyboard
import pyttsx3
import mediapipe as mp

from my_functions import image_process, draw_landmarks, keypoint_extraction
from tensorflow.keras.models import load_model
from transformers import T5Tokenizer, T5ForConditionalGeneration
# ----------------------------------------------------
# TTS SETUP
# ----------------------------------------------------
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    if hasattr(voice, 'languages') and len(voice.languages) > 0:
        try:
            if 'en' in voice.languages[0].decode().lower():
                engine.setProperty('voice', voice.id)
                break
        except:
            continue
    elif 'english' in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

# ----------------------------------------------------
# t5_Translation Setup
# ----------------------------------------------------
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

def flan_translate(gloss_text):
    prompt = (
    f"You are an expert in American Sign Language. Given the following sequence of glosses, "
    f"generate a natural English sentence that conveys the intended meaning. "
    f"Understand the context, rearrange the words as needed, and remove any irrelevant or meaningless glosses: {gloss_text}"
)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    output_ids = flan_model.generate(inputs["input_ids"], max_length=64)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ----------------------------------------------------
# Load LSTM model and prepare labels
# ----------------------------------------------------
DATA_FOLDER = os.path.join('data')
actions = sorted([d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))])
actions = np.array(actions)
print("Loaded actions:", actions)

model = load_model('my_model.h5')

# ----------------------------------------------------
# Runtime Variables
# ----------------------------------------------------
sentence, keypoints, last_prediction = [], [], None
context_output = None
context_spoken = False

# ----------------------------------------------------
# Camera Setup
# ----------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

with mp.solutions.holistic.Holistic(min_detection_confidence=0.75,
                                    min_tracking_confidence=0.75) as holistic:
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            continue

        results = image_process(image, holistic)
        draw_landmarks(image, results)
        keypoints.append(keypoint_extraction(results))

        if len(keypoints) == 10:
            keypoints_array = np.array(keypoints)
            keypoints = []

            prediction = model.predict(keypoints_array[np.newaxis, :, :])
            predicted_class_idx = np.argmax(prediction)
            predicted_class_prob = np.max(prediction)

            if predicted_class_prob > 0.9:
                predicted_sign = actions[predicted_class_idx]
                if predicted_sign != last_prediction:
                    sentence.append(predicted_sign)
                    last_prediction = predicted_sign
                    context_spoken = False

        if len(sentence) > 7:
            sentence = sentence[-7:]

        if keyboard.is_pressed(' '):
            sentence, keypoints, last_prediction = [], [], None
            context_output = None
            context_spoken = False

        if sentence:
            sentence[0] = sentence[0].capitalize()

        if len(sentence) >= 2:
            if (sentence[-1] in string.ascii_letters) and (sentence[-2] in string.ascii_letters):
                sentence[-1] = sentence[-2] + sentence[-1]
                sentence.pop(-2)
                sentence[-1] = sentence[-1].capitalize()

        # Enter to trigger contextual translation
        if keyboard.is_pressed('enter') and sentence:
            gloss_text = ' '.join(sentence).lower().strip()
            context_output = flan_translate(gloss_text)
            sentence = []
            last_prediction = None
            context_spoken = False

        if context_output:
            display_text = context_output
            if not context_spoken:
                engine.say(context_output)
                engine.runAndWait()
                context_spoken = True
        else:
            display_text = ' '.join(sentence)

        try:
            display_text_cv = display_text.encode("ascii", "ignore").decode()
        except:
            display_text_cv = "[Rendering Error]"

        textsize = cv2.getTextSize(display_text_cv, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_X_coord = (image.shape[1] - textsize[0]) // 2
        cv2.putText(image, display_text_cv, (text_X_coord, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Camera', image)
        cv2.waitKey(1)

        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()

 
    