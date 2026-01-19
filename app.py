import os
import cv2
import numpy as np
import mediapipe as mp
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras as k3
from threading import Lock
from keras.layers import TFSMLayer

modelPath = './mp_hand_gesture'
namesPath = os.path.join(modelPath, 'gesture.names')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model = None
hands = None
classNames = []
initLoaded = False
initLock = Lock()

def initialize():
    global model, hands, classNames, initLoaded
    with initLock:
        if initLoaded:
            return
        try:
            mpHands = mp.solutions.hands
            hands = mpHands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )

            model = TFSMLayer(modelPath, call_endpoint="serving_default")

            with open(namesPath, 'r') as f:
                classNames = [x.strip() for x in f.read().splitlines() if x.strip()]

            initLoaded = True

        except Exception as e:
            print(f"Initialization Error: {e}", flush=True)
            initLoaded = False

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/status', methods=['GET'])
def status():
    if not initLoaded:
        initialize()
    return jsonify({
        "status": "online" if initLoaded else "initialization_failed",
        "model_ready": initLoaded,
        "classes_loaded": len(classNames) if initLoaded else 0
    }), 200

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
            "Shiva Lingam, Saying 'I don’t know', Remembering, Acting,\n"
            "Tying around the waist, Embrace, Ringing bells, Peak"
        )

    elif className == "fist":
        details["name"] = "Mushti"

    elif className in ["stop", "live long"]:
        details["name"] = "Patakam"

    return details

@app.route('/recognize_gesture', methods=['POST'])
def recognize_gesture():
    if not initLoaded:
        initialize()
        if not initLoaded:
            return jsonify({"error": "Server failed to initialize resources."}), 503

    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({"error": "Missing 'image_data'"}), 400

        image_b64 = data['image_data']
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]

        missing_padding = len(image_b64) % 4
        if missing_padding:
            image_b64 += "=" * (4 - missing_padding)

    except Exception as e:
        return jsonify({"error": f"Invalid request: {e}"}), 400

    try:
        img_bytes = base64.b64decode(image_b64)
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

    except Exception as e:
        return jsonify({"error": f"Image decode error: {e}"}), 500

    h, w, _ = frame.shape
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frameRGB)

    className = "No Hand Detected"

    if result.multi_hand_landmarks:
        lms = result.multi_hand_landmarks[0]
        points = []

        for lm in lms.landmark:
            points.append([int(lm.x * w), int(lm.y * h)])

        if len(points) != 21:
            return jsonify({"error": "Incomplete landmarks"}), 400

        inp = np.array(points).flatten().reshape(1, -1)
        #pred = model.predict(inp, verbose=0)
        pred = model(inp) 
        # If pred is a dict (some SavedModels return dicts), pick the first value: 
        if isinstance(pred, dict): 
            pred = list(pred.values())[0] 
        # Convert to numpy if needed 
        pred = np.array(pred)
        classID = np.argmax(pred)

        if classID >= len(classNames):
            return jsonify({"error": "Invalid classID"}), 500

        className = classNames[classID]

    if className == "No Hand Detected":
        return jsonify({
            "status": "success",
            "gesture_name": className,
            "mudra_name": None,
            "viniyoga_shloka": None,
            "meaning": None
        }), 200

    details = get_mudra_details(className)

    return jsonify({
        "status": "success",
        "gesture_name": className,
        "mudra_name": details["name"],
        "viniyoga_shloka": details["shloka"],
        "meaning": details["meaning"]
    }), 200

initialize()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
