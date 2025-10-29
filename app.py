import os
import cv2
import numpy as np
import mediapipe as mp
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras as k3

#must install dependencies on deployment server

#relative paths
modelPath = './mp_hand_gesture'
namesPath = os.path.join(modelPath, 'gesture.names')

#initialize flask
app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

#initialize variables
model = None
hands = None
classNames = []
initLoaded = False

def initialize():
    global model, hands, classNames, initLoaded
    try:
        print("STARTING INITIALIZATION", flush=True)
        #initialize MediaPipe
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)

        #load model
        print("LOADING MODEL", flush=True)
        model = k3.models.load_model(modelPath)
        
        print("LOADING CLASS NAMES", flush=True)
        #load class names
        with open(namesPath, 'r') as f:
            classNames = f.read().split('\n')

        #flag: successful initialization and loading
        print("LOADED", flush=True)
        initLoaded = True

    except Exception as e:
        print(f"Error loading resources: {e}")
        initLoaded = False

@app.route('/health', methods=['GET'])
def get_health():
    #health check for the deployment platform to confirm the app is running
    return jsonify({"status": "ok", "app_running": True}), 200

@app.route('/status', methods=['GET'])
def get_status():
    #health status of the API and model readiness
    if not initLoaded:
        initialize()
    return jsonify({
        "status": "online" if initLoaded else "initialization_failed",
        "model_ready": initLoaded
    })

#mudra details

def get_mudra_details(className):
    details = {
        "name": "Mudra not mapped yet",
        "shloka": "Not mapped yet",
        "meaning": "Not mapped yet"
    }

    if className in ["thumbs up", "thumbs down"]:
        details["name"] = "Shikhara"
        details["shloka"] = (
            "Madana kamuka sthambaecha Nishcaya pitrukarmani\n"
            "Oshtra pravishtaroopani Radhana prashnabhavanae\n"
            "Linga nastheethivachanae Samarana katibandakarshana\n"
            "Parirambhavidikrama Gandani nada Shikarayujyatahe Bharatadibi"
        )
        details["meaning"] = (
            "God of love, a Bow, a Pillar, To decide, Making offering to Manes,\n"
            "Lips, To Enter or to pour, Tooth, Questioning,\n"
            "Shiva Lingam, Saying 'I don’t know', An act of remembering, To act, To tie around the waist,\n"
            "To embrace, Ringing of bells, Peak"
        )
    elif className == "fist":
        details["name"] = "Mushti"
    elif className in ["stop", "live long"]:
        details["name"] = "Patakam"

    return details

#API endpoint

@app.route('/recognize_gesture', methods=['POST'])
def recognize_gesture():
    """Receives Base64 image, recognizes gesture, and returns Mudra details."""
    if not initLoaded:
        initialize()
        if not initLoaded:
            return jsonify({"error": "Server failed to initialize and load resources."}), 503

    #receive and validate data from client side (Weebly JavaScript)
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({"error": "No image data provided in the request payload."}), 400

        image_b64 = data['image_data']
        #to remove "data:image/jpeg;base64," prefix, if exists
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]

    except Exception as e:
        return jsonify({"error": f"Invalid JSON or request format: {e}"}), 400

    #base64 Image to OpenCV/NumPy array
    try:
        image_bytes = base64.b64decode(image_b64)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        #if image decoded properly
        if frame is None:
            return jsonify({"error": "Could not decode image data."}), 400

    except Exception as e:
        return jsonify({"error": f"Error in decoding image: {e}"}), 500

    #gesture recognition logic (of OG GC code)

    x, y, c = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    className = 'No Hand Detected' #default

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            #predict gesture using first detected hand
            prediction = model.predict([landmarks], verbose=0)
            classID = np.argmax(prediction)
            className = classNames[classID]
            break

    #fetch mudra details
    mudra_details = get_mudra_details(className)

    #return JSON
    return jsonify({
        "status": "success",
        "gesture_name": className,
        "mudra_name": mudra_details['name'],
        "viniyoga_sloka": mudra_details['sloka'],
        "meaning": mudra_details['meaning']
    })

initialize()
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
