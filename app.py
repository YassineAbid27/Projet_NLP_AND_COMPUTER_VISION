from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import tensorflow as tf
import os

# Configure Flask to serve static files from the "static" folder.
app = Flask(__name__, static_url_path='', static_folder='static')

# Load the trained model once at startup.
model = tf.keras.models.load_model('my_model.h5')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    if not json_data or 'data' not in json_data:
        return jsonify({'error': 'No data provided. Please provide a "data" key with input frames.'}), 400

    try:
        # Convert the received data to a numpy array.
        data = np.array(json_data['data'])
        if data.ndim != 2 or data.shape[0] != 10:
            return jsonify({'error': 'Input data must have shape (10, features).'}), 400

        # Expand dimensions to add the batch dimension (resulting shape: (1, 10, features)).
        input_data = np.expand_dims(data, axis=0)
        preds = model.predict(input_data)
        pred_class = int(np.argmax(preds, axis=1)[0])
        probability = float(np.max(preds))
        
        return jsonify({'predicted_class': pred_class, 'probability': probability})
    except Exception as e:
        app.logger.error("Prediction error: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask server on port 5000.
    # Note: debug=True is fine for development, but use debug=False in production
    app.run(host='0.0.0.0', port=5000, debug=True)