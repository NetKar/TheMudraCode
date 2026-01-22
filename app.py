import os
import cv2
import numpy as np
import mediapipe as mp
import base64
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Lock

modelPath = './mp_hand_gesture'   # folder containing saved_model.pb + variables/
namesPath = os.path.join(modelPath, 'gesture.names')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model = None
infer = None
hands = None
classNames = []
initLoaded = False
initLock = Lock()

def initialize():
    global model, infer, hands, classNames, initLoaded
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

            # Load SavedModel directly
            model = tf.saved_model.load(modelPath)
            infer = model.signatures["serving_default"]

            with open(namesPath, 'r') as f:
                classNames = [x.strip() for x in f.read().splitlines() if x.strip()]

            initLoaded = True
            print("Initialization successful", flush=True)

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

    if className in ["thumbs up", "thumbs down", "call me"]:
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
        details["shloka"] = (
            "Sthiram kachagraha daartya\n"
            "Vasthuvadeenam cha dharana\n"
            "Mallianam yudha bhavae cha\n"
            "Mushti hastanam yujyathae"
        )
        details["meaning"] = (
            "Steadiness, Grasping one’s hair, Courage\n"
            "Holding objects\n"
            "Fighting mood of wrestlers\n"
            "Thus, these are the uses of Mushti hasta"
        )

    elif className in ["stop", "live long"]:
        details["name"] = "Pathaka"
        details["shloka"] = (
            "Natyarambhae vaarivahae vanai vastu nishedhanae\n"
            "Kuchasthalae nishayam cha nadyam amaramandalae\n"
            "Turangae khandanae vayo shayanae gamanodyamae\n"
            "Pratapae cha prasadae cha chandrikayam ghana tapae\n"
            "Kavaadapaatanae sapta vibhaktyarthae tharangae\n"
            "Veedi pravesha bhavepi samatvae cha angaragakae\n"
            "Aatmarthae shapathae chapae thooshnim bhava nidharshanae\n"
            "Thaalapatrae cha kheitae cha dravyadi sparshanae thatha\n"
            "Aashirvadae kriyayam cha nrupa sreshtasya darshanae\n"
            "Thatra thatreti vachanae sindhyo cha sukruthi kramae\n"
            "Sambhodhanae purogaepae khadga roopasya dharanae\n"
            "Masae samvathsarae varsha dinae sammarjanae thatha\n"
            "Yevam artheshu yujanthae pataka hasta bhavanah"
        )
        details["meaning"] = (
            "Beginning of dance, Rain clouds, Forest, To deny or avoid\n"
            "Bosom, Night, River, Heaven\n"
            "Horse, Cutting, Wind, Sleeping, Walking or movement\n"
            "Showing power, Blessing, Moonlight, Strong sunlight\n"
            "Opening or closing a door, Denoting the seven grammatical cases, Waves\n"
            "Entering a street, Equality, Applying sandal paste or massaging the body\n"
            "Oneself, Taking an oath, Silence or secret indication\n"
            "Palm leaf (writing a letter), Shield, Touching objects\n"
            "Blessing action, Presence of a powerful king\n"
            "Saying ‘this’ or ‘that’, Ocean, Being virtuous or good deeds\n"
            "Addressing someone, Moving forward, Holding the form of a sword\n"
            "Month, Year, Rainy day, Sweeping\n"
            "Thus, Patāka hasta is used in all these meanings"
        )

    elif className == "okay":
        details["name"] = "Hamsasya"
        details["shloka"] = (
            "Mangalyasutra bandhe cha api upadeshae\n"
            "Vinishchayae romanche mouktikadou cha\n"
            "Chitra samlekhanae thatha\n"
            "Damshathu cha jala bindou cha\n"
            "Deepa varti prasaranae\n"
            "Nikashae shodhanae mallikadou cha\n"
            "Rekha valekhanae malayam vahanae\n"
            "Soham bhavanae cha roopakae\n"
            "Naasteeti vachanae cha api nikashanam cha bhavane\n"
            "Kruta krutyepi hamsasyaha eerito bharatagamae"

        )
        details["meaning"] = (
            "Tying the sacred marriage thread, Giving advice\n"
            "Decision making, Excitement, Pearls and other precious stones\n"
            "Drawing or sketching\n"
            "Fly, Drop of water\n"
            "Spreading the wick of a lamp\n"
            "Polishing, Searching, Jasmine and other flowers\n"
            "Drawing a line, Holding a garland\n"
            "Expressing ‘I am Brahma’, Dramatic representation\n"
            "Saying ‘No’, Looking at a polished object\n"
            "Thus are the applications of Hamsasya hasta as explained by Bharata"
        )

    elif className == "smile":
        details["name"] = "Chandrakalaa"
        details["shloka"] = (
            "Chandray mukhay cha pradayshay tanmatra akara vastuni\n"
            "Shivasya makutey ganga nadyam cha lagudey pi cha\n"
            "Esha chandrakala chaiva viniyojana vidhiyatey"
        )
        details["meaning"] = (
            "Moon, Face, Index of measure (distance between thumb and index finger)\n"
            "Objects of similar crescent-like shape\n"
            "The crescent moon on Lord Shiva’s head\n"
            "The river Ganga, A club or weapon\n"
            "These are the uses of Chandrakala hasta\n"
            "Thus is the viniyoga as prescribed"
        )
    
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

        inp = np.array(points).flatten().reshape(1, -1).astype(np.float32)

        # Run inference using SavedModel signature
        pred = infer(tf.constant(inp))
        pred = list(pred.values())[0].numpy()

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
