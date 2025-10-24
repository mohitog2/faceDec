from flask import Flask, render_template, request, jsonify
from fer import FER
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
detector = FER(mtcnn=True)

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))


@app.route('/')
def index():
    return render_template('index.html')  # your HTML file

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get base64 image from frontend
        data = request.json['image'].split(',')[1]
        img_bytes = base64.b64decode(data)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect emotions
        result = detector.detect_emotions(img)
        emotion = "neutral"
        if result:
            emotions = result[0]["emotions"]
            emotion = max(emotions, key=emotions.get)
        
        return jsonify({"emotion": emotion})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
