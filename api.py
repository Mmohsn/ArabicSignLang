
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import mediapipe as mp
import joblib
from PIL import Image, ImageDraw, ImageFont
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display
from io import BytesIO


categories=[
["fine",'بخير'],
["hello","مرحبا, كيف حالك؟"],
["stop",'قف'],
["yes",'نعم']]
words = []
sequence = ''
fontFile = "Sahel.ttf"
font = ImageFont.truetype(fontFile, 40)
app = Flask(__name__)

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1,detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, 
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []
        bbox = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            xList = []
            yList = []
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
        return lmlist, bbox
    
    
@app.route('/predict', methods=['POST'])
def predict_gesture():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    file = request.files['frame']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    model = joblib.load('hand_gesture_model.pkl')
    detector = handDetector()

    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img)
    if bbox:
        x, y, x2, y2 = bbox
        hand_img = img[y:y2, x:x2]
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        hand_img = cv2.resize(hand_img, (64, 64))
        hand_img_flat = hand_img.flatten().reshape(1, -1)
        prediction = model.predict(hand_img_flat)
        print(prediction[0])
        probability = model.predict_proba(hand_img_flat).max()
        for category in categories:
                if category[0] == prediction[0]:
                    sequence = category[1]
        reshaped_text = arabic_reshaper.reshape(sequence)   
        bidi_text = get_display(reshaped_text) 
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((100, 300), bidi_text, (0,0,0), font=font,align="center")
        img = np.array(img_pil)
        # Draw bounding box and prediction text on the image
        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{prediction[0]} ({probability:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert image to PNG and return as response
        _, buffer = cv2.imencode('.png', img)
        return send_file(BytesIO(buffer), mimetype='image/png'), 200

    return jsonify({'error': 'No hand detected'}), 404

# This block is required for the Flask application to be run directly
if __name__ == '__main__':
    app.run(debug=True)

