# Sign Language Translator 🤟

A real-time American Sign Language (ASL) translator using MediaPipe for hand tracking, LSTM neural networks for sign recognition, and T5 language model for contextual translation to natural English.

## Features

- ✨ Real-time hand landmark detection using MediaPipe
- 🧠 LSTM-based sign language recognition
- 🗣️ T5 model for gloss-to-English translation
- 🔊 Text-to-speech output
- 📊 Custom dataset collection and training
- 🌐 Flask web API for predictions

## System Requirements

- Python 3.10
- Anaconda (recommended)
- Webcam
- Windows/Linux/MacOS

## Installation Guide

### Step 1: Create Anaconda Environment

Open Anaconda Prompt (or terminal) and create a new environment:

```bash
conda create -n sign_language python=3.10 -y
```

Activate the environment:

```bash
conda activate sign_language
```

### Step 2: Clone/Download the Project

Navigate to your desired directory and clone or download this project:

```bash
cd C:\Users\YourUsername\Desktop
# If you have the project folder, navigate into it
cd "Sign-Language-Translator"
```

### Step 3: Install Dependencies

Install all required packages using the requirements file:

```bash
pip install -r requirements.txt
```

**Important:** If you encounter any errors, install packages in this specific order:

```bash
pip install numpy==1.24.3
pip install protobuf==3.20.3
pip install tensorflow==2.13.0
pip install mediapipe==0.10.5
pip install typing_extensions>=4.8.0
pip install -r requirements.txt
```

### Step 4: Verify Installation

Test if all packages are installed correctly:

```bash
python -c "import mediapipe as mp; import tensorflow as tf; from transformers import T5Tokenizer; print('✓ All packages installed successfully!')"
```

## Usage Guide

### Option A: Using Pre-trained Model (If Available)

If you have a pre-trained `my_model.h5` file, skip to **Step 3**.

### Option B: Train Your Own Model

#### Step 1: Collect Training Data

First, decide which ASL signs you want to recognize. Edit `data_collection.py`:

```python
# Example: Collect data for three signs
actions = np.array(['I'])  # Start with one sign
sequences = 30  # Number of sequences to record
frames = 10     # Frames per sequence
```

Run the data collection script:

```bash
python data_collection.py
```

**Instructions:**
1. Position yourself in front of the webcam
2. Press **SPACE** to start recording each sequence
3. Perform the sign while the camera records 10 frames
4. Repeat 30 times for each sign
5. Close the window when done

**Repeat this process for each sign** by changing the `actions` array:

```python
# First run
actions = np.array(['I'])

# Second run (after completing first)
actions = np.array(['NEED'])

# Third run
actions = np.array(['HELP'])
```

#### Step 2: Train the Model

After collecting data for all your signs, train the model:

```bash
python model.py
```

This will:
- Load all collected sign data from the `data` folder
- Split into training/testing sets
- Train an LSTM model (100 epochs)
- Save the trained model as `my_model.h5`
- Display accuracy metrics

**Training time:** ~5-15 minutes depending on your CPU/GPU and dataset size.

#### Step 3: Run the Real-time Translator

Start the sign language translator:

```bash
python main.py
```

**How to use:**
1. The webcam window will open showing your hands with landmarks
2. Perform signs in front of the camera
3. Detected signs appear at the bottom of the screen
4. Press **ENTER** to translate the gloss sequence to natural English
5. The T5 model will speak the translated sentence
6. Press **SPACE** to clear the sentence and start over
7. Close the window or press ESC to exit

**Controls:**
- **SPACE**: Clear current sentence
- **ENTER**: Translate gloss to English and speak
- **ESC/Close Window**: Exit application

## Project Structure

```
Sign-Language-Translator/
│
├── data/                          # Collected training data (auto-created)
│   ├── I/
│   │   ├── 0/                    # Sequence 0
│   │   │   ├── 0.npy             # Frame 0 landmarks
│   │   │   ├── 1.npy
│   │   │   └── ...
│   │   ├── 1/                    # Sequence 1
│   │   └── ...
│   ├── NEED/
│   └── HELP/
│
├── static/                        # Flask static files (optional)
│   └── index.html
│
├── data_collection.py            # Collect training data from webcam
├── data_preprocessing.py         # Preprocess large datasets (optional)
├── model.py                      # Train LSTM model
├── main.py                       # Real-time translator with T5
├── app.py                        # Flask web API
├── my_functions.py               # Helper functions
├── requirements.txt              # Python dependencies
├── my_model.h5                   # Trained model (generated)
└── README.md                     # This file
```

## Advanced Usage

### Using the Flask Web API

Start the Flask server:

```bash
python app.py
```

The API will be available at `http://localhost:5000`

**API Endpoint:**

```
POST /predict
Content-Type: application/json

{
  "data": [[...], [...], ...]  // 10 frames of landmark data
}

Response:
{
  "predicted_class": 0,
  "probability": 0.95
}
```

### Processing Large Datasets

If you have a large ASL dataset (like Kaggle ASL datasets), use the preprocessing script:

1. Update the paths in `data_preprocessing.py`:
```python
TRAIN_CSV = r"path/to/train.csv"
TRAIN_LANDMARK_FILES = r"path/to/train_landmark_files"
OUTPUT_DIR = r"path/to/output"
```

2. Run the preprocessing:
```bash
python data_preprocessing.py
```

## Recommended ASL Phrases for Training

Start with simple, practical phrases:

### Beginner (3-4 signs):
- **I NEED HELP**
- **WHERE YOU LIVE**
- **THANK YOU**

### Intermediate (4-5 signs):
- **YESTERDAY I GO STORE**
- **TOMORROW I WORK**
- **WEATHER TODAY COLD**

### Advanced (5+ signs):
- **I NOT UNDERSTAND YOU**
- **PLEASE HELP ME LEARN**

**Tip:** Choose signs with distinct hand shapes for better recognition accuracy.

## Troubleshooting

### Issue: "Cannot access camera"
- **Solution:** Check if another application is using the webcam
- Try changing camera index in the code: `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)`

### Issue: "No module named 'mediapipe'"
- **Solution:** Reinstall mediapipe:
```bash
pip uninstall mediapipe
pip install mediapipe==0.10.5
```

### Issue: Protobuf or TensorFlow errors
- **Solution:** Reinstall with exact versions:
```bash
pip uninstall tensorflow mediapipe protobuf numpy
pip install numpy==1.24.3
pip install protobuf==3.20.3
pip install tensorflow==2.13.0
pip install mediapipe==0.10.5
```

### Issue: Low prediction accuracy
- **Solutions:**
  - Collect more training sequences (increase from 30 to 50+)
  - Ensure consistent lighting conditions
  - Perform signs clearly and consistently
  - Add more training epochs in `model.py`

### Issue: T5 model not translating well
- **Solution:** The T5 model works best with 3+ word sequences
- Use clear ASL glosses (uppercase words)
- Ensure signs are in proper ASL grammar order

## Performance Tips

### For Better Accuracy:
1. **Lighting:** Use good, consistent lighting
2. **Background:** Use a plain background
3. **Hand position:** Keep hands clearly visible in frame
4. **Consistency:** Perform signs the same way during collection and testing
5. **Data quantity:** Collect 50+ sequences per sign if possible

### For Faster Training:
- Use GPU if available (CUDA-enabled GPU)
- Reduce epochs in `model.py` for testing (e.g., 50 instead of 100)
- Use smaller dataset for initial testing

## Technical Details

### Model Architecture:
- **Input:** 10 frames × 126 features (21 landmarks × 3 coords × 2 hands)
- **LSTM Layers:** 32 → 64 → 32 units
- **Output:** Softmax classification over N signs
- **Optimizer:** Adam
- **Loss:** Categorical crossentropy

### T5 Translation:
- **Model:** google/flan-t5-large
- **Task:** Gloss-to-English translation
- **Input:** ASL gloss sequence (e.g., "I NEED HELP")
- **Output:** Natural English sentence (e.g., "I need help")

## Dependencies

- **numpy==1.24.3** - Array operations
- **opencv-python==4.8.1.78** - Computer vision
- **mediapipe==0.10.5** - Hand landmark detection
- **tensorflow==2.13.0** - Deep learning framework
- **protobuf==3.20.3** - Data serialization
- **transformers>=4.35.0** - T5 model
- **torch>=2.1.0** - PyTorch backend
- **scikit-learn>=1.3.0** - ML utilities
- **pyttsx3>=2.90** - Text-to-speech
- **keyboard>=0.13.5** - Keyboard input
- **flask>=3.0.0** - Web framework
- **pandas>=2.0.0** - Data manipulation

## License

This project is for educational purposes.

## Credits

- **MediaPipe** - Google's hand tracking solution
- **TensorFlow** - Deep learning framework
- **Hugging Face Transformers** - T5 model
- **OpenCV** - Computer vision library

## Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Verify all dependencies are correctly installed
3. Ensure Python 3.10 is being used
4. Check that your webcam is working

## Future Improvements

- [ ] Add more ASL signs
- [ ] Implement sentence context memory
- [ ] Add GUI interface
- [ ] Support for facial expressions
- [ ] Real-time continuous translation

---

**Happy Signing! 🤟**

For questions or contributions, please open an issue or pull request.