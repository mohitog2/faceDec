from flask import Flask, render_template, Response, jsonify
import cv2
from fer import FER
from collections import deque
import time

app = Flask(__name__)

# Initialize FER detector and camera
detector = FER(mtcnn=True)
camera = cv2.VideoCapture(0)

prev_emotion = None
emotion_history = deque(maxlen=10)
last_update_time = time.time()

# Emotion → image URLs
emotion_images = {
    "happy": "https://th.bing.com/th/id/OIP.b1k2rrW5wGek6O6_gWEtoAHaHZ?w=183&h=182&c=7&r=0&o=7&cb=12&dpr=1.3&pid=1.7&rm=3.jpg",
    "sad": "https://media.stickerswiki.app/cryingcatdeluxe/1592368.512.webp",
    "angry": "https://i.pinimg.com/736x/e7/40/03/e7400321d9d52fe28b88a1fc91b4bcc7.jpg",
    "neutral": "https://th.bing.com/th/id/OIP.JFU_67aUxz54H0RONq1BZAHaHB?w=179&h=180&c=7&r=0&o=7&cb=12&dpr=1.3&pid=1.7&rm=3.jpg"
}

def gen_frames():
    global prev_emotion, last_update_time

    while True:
        success, frame = camera.read()
        if not success:
            break

        result = detector.top_emotion(frame)
        if result:
            emotion, score = result
            emotion_history.append(emotion)

            # Check if the last 5 detections are all same
            if len(emotion_history) == 10 and len(set(list(emotion_history)[-5:])) == 1:
                stable_emotion = emotion_history[-1]

                # Only update if new emotion AND 2 seconds have passed
                if stable_emotion != prev_emotion and (time.time() - last_update_time > 2):
                    prev_emotion = stable_emotion
                    last_update_time = time.time()
                    yield f"data:{stable_emotion}\n\n"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='text/event-stream')

@app.route('/get_image/<emotion>')
def get_image(emotion):
    url = emotion_images.get(emotion, emotion_images["neutral"])
    return jsonify({"url": url})

if __name__ == '__main__':
    app.run(debug=True)

