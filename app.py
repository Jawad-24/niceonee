from flask import Flask, render_template, Response, request, jsonify
import cv2
from ultralytics import YOLO
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd

# Initialize Firebase Admin SDK
cred = credentials.Certificate('final-ce293-firebase-adminsdk-2esv3-7bd17c32d2.json')  # Replace with the path to your service account key file
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://final-ce293-default-rtdb.firebaseio.com/'  # Replace with your Firebase Realtime Database URL
})

app = Flask(__name__)

# Ensure the upload directory exists
os.makedirs('./static/uploads', exist_ok=True)

# Load the YOLO model
model = YOLO('')
class_labels = {0: 'kiri'}

@app.route('/')
def index():
    return render_template('forecasting.html')

@app.route('/restaurantpage')
def restaurantpage():
    return render_template('restaurantpage.html')

@app.route('/inventory')
def inventory():
    ref = db.reference('/object_counts')
    data = ref.get()
    df = pd.DataFrame(list(data.values()))
    df = df.groupby('product_id')[['count']].sum().reset_index()
    df['product_id'] = df['product_id'].apply(lambda x: class_labels[x])
    
    # Convert data to a list of dictionaries for rendering
    inventory_items = df.to_dict(orient='records')
    
    return render_template('Inventory.html', inventory_items=inventory_items)

@app.route('/get_inventory_data', methods=['GET'])
def get_inventory_data():
    # Fetch inventory data from Firebase
    ref = db.reference('object_counts')
    data = ref.get()
    if data is None:
        data = {}
    return jsonify(data)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    red_line = int(frame_height * 1 / 3)
    blue_line = int(frame_height * 2 / 3)
    text_color = (255, 255, 255)
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)
    green_color = (0, 255, 0)
    count = 0
    passed_red_line = False
    passed_blue_line = False
    is_in = True

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame)

        cv2.line(frame, (0, red_line), (frame_width, red_line), red_color, 3)
        cv2.line(frame, (0, blue_line), (frame_width, blue_line), blue_color, 3)

        for info in results:
            if len(info.boxes.conf) != 0 and info.boxes.conf[0] > 0.5:  # Adjusted threshold
                class_id = int(info.boxes.cls[0])
                label = class_labels[class_id]
                conf = info.boxes.conf[0]
                x1, y1 = int(info.boxes.xyxy[0][0]), int(info.boxes.xyxy[0][1])
                x2, y2 = int(info.boxes.xyxy[0][2]), int(info.boxes.xyxy[0][3])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, f'{label} detected {conf:.2f}', (cx + 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

                if cy < (red_line + 20) and cy > (red_line - 20):
                    passed_red_line = True
                    if not passed_blue_line:
                        is_in = False
                if cy < (blue_line + 20) and cy > (blue_line - 20):
                    passed_blue_line = True

                if passed_blue_line and passed_red_line:
                    if is_in:
                        count = 1
                    else:
                        count = -1
                    
                    new_row = {
                        'product_id': class_id,
                        'is_add': is_in,
                        'count': count,
                        'date': datetime.now().isoformat()
                    }
                    db.reference('object_counts').push(new_row)
                    passed_red_line, passed_blue_line = False, False
                    is_in = True

        cv2.putText(frame, f'COUNT: {count}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 1, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)
