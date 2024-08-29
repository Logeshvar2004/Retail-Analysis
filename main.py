import cv2
import datetime
import imutils
import numpy as np
import csv
import json
from centroidtracker import CentroidTracker
from ultralytics import YOLO
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

age_net = cv2.dnn.readNetFromCaffe('Retail\\age_deploy.prototxt', 'Retail\\age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('Retail\\gender_deploy.prototxt', 'Retail\\gender_net.caffemodel')

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression: {}".format(e))
        return []

def load_yolo_model(weights_path):
    model = YOLO(weights_path)
    return model

def detect_age_gender(frame, bbox):
    x1, y1, x2, y2 = bbox
    face = frame[y1:y2, x1:x2]
    
    if face.shape[0] == 0 or face.shape[1] == 0:
        return None, None

    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104, 117, 123))

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    age = age_list[age_preds[0].argmax()]

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = 'Male' if gender_preds[0].argmax() == 0 else 'Female'

    return age, gender

def generate_unique_id(existing_ids):
    new_id = max(existing_ids) + 1 if existing_ids else 1
    while new_id in existing_ids:
        new_id += 1
    return new_id

def load_zones_from_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    zones = {}
    for zone in data:
        name = zone["name"]
        points = np.array(zone["points"], dtype=np.int32)
        polygon = Polygon(points)
        zones[name] = polygon
    return zones

def is_inside_zone(point, zones):
    p = Point(point)
    for name, polygon in zones.items():
        if polygon.contains(p):
            return name
    return None

def main():
    weights_path = "yolov8n.pt"
    model = load_yolo_model(weights_path)
    
    cap = cv2.VideoCapture('Retail\\2.mp4')

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0

    tracker = CentroidTracker(maxDisappeared=80)
    object_id_list = {}
    dtime = {}
    dwell_time = {}

    zones = load_zones_from_json('Retail\\zone.json')

    with open('detections.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Frame', 'Object ID', 'Age', 'Gender', 'Dwell Time', 'Zone'])

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame = imutils.resize(frame, width=640)
            total_frames += 1

            # Perform detection
            results = model(frame)
            detections = results[0].boxes

            rects = []
            for detection in detections:
                if detection.conf[0] > 0.5 and detection.cls[0] == 0:
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])
                    rects.append([x1, y1, x2, y2])

            # Apply non-max suppression
            rects = np.array(rects)
            rects = non_max_suppression_fast(rects, 0.4)

            # Draw zones
            for name, polygon in zones.items():
                points = np.array(polygon.exterior.coords, dtype=np.int32)
                cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, name, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            objects = {}
            for (objectId, bbox) in tracker.update(rects).items():
                if len(bbox) == 2:
                    cX, cY = bbox
                else:
                    print(f"Skipping object {objectId} due to unexpected bbox format: {bbox}")
                    continue

                # Check if the centroid is inside any zone
                zone_detected = is_inside_zone((cX, cY), zones)

                if zone_detected:
                    age, gender = detect_age_gender(frame, [int(cX - 15), int(cY - 15), int(cX + 15), int(cY + 15)])
                    age_text = f"Age: {age}" if age is not None else "Age: Unknown"
                    gender_text = f"Gender: {gender}" if gender is not None else "Gender: Unknown"

                    if objectId not in object_id_list:
                        unique_id = generate_unique_id(set(object_id_list.values()))
                        object_id_list[objectId] = unique_id
                        dtime[unique_id] = datetime.datetime.now()
                        dwell_time[unique_id] = 0
                    else:
                        unique_id = object_id_list[objectId]
                        curr_time = datetime.datetime.now()
                        old_time = dtime[unique_id]
                        time_diff = curr_time - old_time
                        dtime[unique_id] = curr_time
                        sec = time_diff.total_seconds()
                        dwell_time[unique_id] += sec

                    cv2.rectangle(frame, (int(cX - 15), int(cY - 15)), (int(cX + 15), int(cY + 15)), (0, 0, 255), 2)
                    text = "{}|{}|{}|{}|{}".format(unique_id, int(dwell_time[unique_id]), age_text, gender_text, zone_detected)
                    cv2.putText(frame, text, (int(cX - 15), int(cY - 15) - 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 0, 255), 1)

                    # Write to CSV file
                    csv_writer.writerow([total_frames, unique_id, age if age is not None else 'Unknown', gender if gender is not None else 'Unknown', int(dwell_time[unique_id]), zone_detected])

            # Show the frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
