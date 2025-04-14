import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import time
import os
import json
from shapely.geometry import Point, Polygon

model = YOLO('yolov8n.pt')
video_path = r'Retail\2.mp4'
with open('D:\Coding\python\Retail\zone2.json', 'r') as f:
    zone_data = json.load(f)

# Convert JSON zone points to Polygon(s)
zones = {}
for zone in zone_data:
    polygon_points = [tuple(point) for point in zone['points']]
    zones[zone['name']] = Polygon(polygon_points)

# Load age and gender models
age_net = cv2.dnn.readNetFromCaffe('Retail\\Models\\age_deploy.prototxt', 'Retail\\Models\\age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('Retail\\Models\\gender_deploy.prototxt', 'Retail\\Models\\gender_net.caffemodel')

# Define age and gender lists
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Function to calculate the centroid of a bounding box
def calculate_centroid(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy

# Function to check which zone a point is in
def get_zone(point):
    point_obj = Point(point)
    for zone_name, zone_polygon in zones.items():
        if zone_polygon.contains(point_obj):
            return zone_name
    return None

# Initialize Kalman Filter parameters
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])
kf.R *= 10.  # Measurement uncertainty
kf.P *= 10.  # Initial state uncertainty
kf.Q *= 0.01  # Process uncertainty

# Initialize a dictionary to store object IDs and their Kalman Filters
object_kalman_filters = {}
object_centroids = defaultdict(list)

# Load existing IDs from CSV if it exists
csv_path = 'output.csv'
if os.path.exists(csv_path):
    existing_data = pd.read_csv(csv_path)
    if not existing_data.empty:
        max_id = existing_data['ID'].max()
        next_object_id = max_id + 1
    else:
        next_object_id = 0
else:
    next_object_id = 0

# Dictionary to store entry times and positions of objects
entry_times = {}
dwell_times = {}
positions = {}
object_zones = {}  # Track current zone for each object
zone_entry_times = {}  # Track when an object entered a zone
zone_dwell_times = defaultdict(lambda: defaultdict(float))  # Track dwell time in each zone

# Data for CSV
data = []

# Zone-specific statistics
zone_stats = {zone_name: {'visitors': set(), 'avg_dwell_time': 0, 'male_count': 0, 'female_count': 0, 'age_groups': defaultdict(int)} 
              for zone_name in zones.keys()}

cap = cv2.VideoCapture(video_path)

ret = True
while ret:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    new_centroids = []
    new_boxes = []
    new_ids = []
    new_zones = []  # Store zone information for each detection

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                label = box.cls.item()
                if label == 0:  # Assuming 'person' is the label for people
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = box.conf.item()
                    centroid = calculate_centroid((x1, y1, x2, y2))
                    
                    # Check which zone the person is in
                    zone_name = get_zone(centroid)
                    if zone_name:  # If the person is in any zone
                        new_centroids.append(centroid)
                        new_boxes.append((x1, y1, x2, y2))
                        new_zones.append(zone_name)

    # Predicted centroids based on Kalman Filters
    predicted_centroids = {}

    for object_id, kf in object_kalman_filters.items():
        kf.predict()
        predicted_centroids[object_id] = (int(kf.x[0]), int(kf.x[1]))

    # Match existing objects to new detections using Hungarian Algorithm
    if len(predicted_centroids) > 0 and len(new_centroids) > 0:
        D = np.zeros((len(predicted_centroids), len(new_centroids)))

        for i, (object_id, pred_centroid) in enumerate(predicted_centroids.items()):
            for j, new_centroid in enumerate(new_centroids):
                D[i, j] = np.linalg.norm(np.array(pred_centroid) - np.array(new_centroid))

        row_ind, col_ind = linear_sum_assignment(D)

        used_rows = set()
        used_cols = set()

        for row, col in zip(row_ind, col_ind):
            if D[row, col] < 50:  # Distance threshold for matching
                object_id = list(predicted_centroids.keys())[row]
                object_centroids[object_id].append(new_centroids[col])
                object_kalman_filters[object_id].update(np.array([new_centroids[col][0], new_centroids[col][1]]).reshape(-1, 1))
                
                new_ids.append(object_id)
                current_zone = new_zones[col]
                
                # Check if zone changed
                if object_id in object_zones and object_zones[object_id] != current_zone:
                    # Calculate dwell time in previous zone
                    if object_id in zone_entry_times and object_zones[object_id] in zone_entry_times[object_id]:
                        prev_zone = object_zones[object_id]
                        dwell_in_zone = time.time() - zone_entry_times[object_id][prev_zone]
                        zone_dwell_times[object_id][prev_zone] += dwell_in_zone
                    
                    # Set new zone entry time
                    if object_id not in zone_entry_times:
                        zone_entry_times[object_id] = {}
                    zone_entry_times[object_id][current_zone] = time.time()
                
                # Initialize zone entry time if new to this zone
                elif object_id not in object_zones or (object_id not in zone_entry_times or current_zone not in zone_entry_times[object_id]):
                    if object_id not in zone_entry_times:
                        zone_entry_times[object_id] = {}
                    zone_entry_times[object_id][current_zone] = time.time()
                
                # Update current zone
                object_zones[object_id] = current_zone

                used_rows.add(row)
                used_cols.add(col)

        unused_rows = set(range(D.shape[0])).difference(used_rows)
        unused_cols = set(range(D.shape[1])).difference(used_cols)

        for col in unused_cols:
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.x[:2] = np.array(new_centroids[col], dtype=np.float32).reshape(-1, 1)
            kf.P *= 10.
            object_kalman_filters[next_object_id] = kf
            object_centroids[next_object_id] = [new_centroids[col]]
            
            new_ids.append(next_object_id)
            current_zone = new_zones[col]
            
            # Record entry time and position
            entry_times[next_object_id] = time.time()
            positions[next_object_id] = new_centroids[col]
            
            # Initialize zone tracking for new object
            object_zones[next_object_id] = current_zone
            zone_entry_times[next_object_id] = {current_zone: time.time()}
            
            next_object_id += 1

    else:
        for i in range(len(new_centroids)):
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.x[:2] = np.array(new_centroids[i], dtype=np.float32).reshape(-1, 1)
            kf.P *= 10.
            object_kalman_filters[next_object_id] = kf
            object_centroids[next_object_id] = [new_centroids[i]]
            
            new_ids.append(next_object_id)
            current_zone = new_zones[i]
            
            # Record entry time and position
            entry_times[next_object_id] = time.time()
            positions[next_object_id] = new_centroids[i]
            
            # Initialize zone tracking for new object
            object_zones[next_object_id] = current_zone
            zone_entry_times[next_object_id] = {current_zone: time.time()}
            
            next_object_id += 1

    # Process each detected person
    for box, object_id, i in zip(new_boxes, new_ids, range(len(new_zones))):
        x1, y1, x2, y2 = box
        current_zone = new_zones[i]
        
        # Add to zone visitors
        zone_stats[current_zone]['visitors'].add(object_id)
        
        # Extract face for gender/age prediction
        face = frame[y1:y2, x1:x2]
        if face.size > 0:  # Make sure face is not empty
            # Prepare face for age and gender prediction
            try:
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                
                # Predict gender
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                
                # Update gender count in zone stats if not already counted
                if gender == 'Male' and object_id not in zone_stats[current_zone].get('male_tracked', set()):
                    zone_stats[current_zone]['male_count'] += 1
                    if 'male_tracked' not in zone_stats[current_zone]:
                        zone_stats[current_zone]['male_tracked'] = set()
                    zone_stats[current_zone]['male_tracked'].add(object_id)
                elif gender == 'Female' and object_id not in zone_stats[current_zone].get('female_tracked', set()):
                    zone_stats[current_zone]['female_count'] += 1
                    if 'female_tracked' not in zone_stats[current_zone]:
                        zone_stats[current_zone]['female_tracked'] = set()
                    zone_stats[current_zone]['female_tracked'].add(object_id)
                
                # Predict age
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                
                # Update age group count in zone stats
                if object_id not in zone_stats[current_zone].get('age_tracked', {}).get(age, set()):
                    zone_stats[current_zone]['age_groups'][age] += 1
                    if 'age_tracked' not in zone_stats[current_zone]:
                        zone_stats[current_zone]['age_tracked'] = {}
                    if age not in zone_stats[current_zone]['age_tracked']:
                        zone_stats[current_zone]['age_tracked'][age] = set()
                    zone_stats[current_zone]['age_tracked'][age].add(object_id)
                
            except Exception as e:
                gender = "Unknown"
                age = "Unknown"
                print(f"Error processing face: {e}")
        else:
            gender = "Unknown"
            age = "Unknown"
        
        # Calculate dwell time in current zone
        if object_id in zone_entry_times and current_zone in zone_entry_times[object_id]:
            current_dwell = time.time() - zone_entry_times[object_id][current_zone]
        else:
            current_dwell = 0
        
        # Total dwell time across all zones
        total_dwell = time.time() - entry_times[object_id]
        dwell_times[object_id] = total_dwell
        
        # Check if the person is standing in the same place
        if len(object_centroids[object_id]) > 1:
            initial_position = np.array(object_centroids[object_id][0])
            current_position = np.array(object_centroids[object_id][-1])
            distance_moved = np.linalg.norm(initial_position - current_position)
            
            # Record data for CSV if they've been standing still
            if distance_moved < 20 and total_dwell > 5:  # Threshold distance and dwell time
                data.append([object_id, gender, age, total_dwell, current_zone, current_dwell])
        
        # Draw bounding box and labels
        label = f"ID: {object_id}, {gender}, {age}, Zone: {current_zone}"
        dwell_label = f"Zone time: {current_dwell:.2f}s, Total: {total_dwell:.2f}s"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, dwell_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw zone polygons on the frame
    for zone_name, zone_polygon in zones.items():
        pts = np.array(zone_polygon.exterior.coords, np.int32)
        pts = pts.reshape((-1, 1, 2))
        color = (0, 0, 255)  # Red color for zones
        cv2.polylines(frame, [pts], True, color, 2)
        # Find a good position for the zone label (average of points)
        x_avg = int(np.mean(pts[:, 0, 0]))
        y_avg = int(np.mean(pts[:, 0, 1]))
        cv2.putText(frame, zone_name, (x_avg, y_avg), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Update zone statistics display
    y_pos = 30
    for zone_name, stats in zone_stats.items():
        # Calculate average dwell time for the zone
        visitor_count = len(stats['visitors'])
        if visitor_count > 0:
            total_zone_dwell = sum(sum(zone_dwell_times[visitor_id].get(zone_name, 0) for visitor_id in stats['visitors']) 
                                   for zone_name in zones.keys())
            avg_dwell = total_zone_dwell / visitor_count if visitor_count > 0 else 0
            stats['avg_dwell_time'] = avg_dwell
        
        # Display zone statistics on frame
        cv2.putText(frame, f"Zone: {zone_name}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 20
        cv2.putText(frame, f"Visitors: {visitor_count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 20
        cv2.putText(frame, f"Avg Dwell: {stats['avg_dwell_time']:.2f}s", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 20
        cv2.putText(frame, f"Male: {stats['male_count']}, Female: {stats['female_count']}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 30
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calculate final zone statistics
for zone_name, stats in zone_stats.items():
    visitor_count = len(stats['visitors'])
    if visitor_count > 0:
        total_zone_dwell = 0
        for visitor_id in stats['visitors']:
            if visitor_id in zone_dwell_times and zone_name in zone_dwell_times[visitor_id]:
                total_zone_dwell += zone_dwell_times[visitor_id][zone_name]
        stats['avg_dwell_time'] = total_zone_dwell / visitor_count if visitor_count > 0 else 0

# Print final zone statistics
print("\nZone Statistics:")
for zone_name, stats in zone_stats.items():
    print(f"\nZone: {zone_name}")
    print(f"Total Visitors: {len(stats['visitors'])}")
    print(f"Average Dwell Time: {stats['avg_dwell_time']:.2f} seconds")
    print(f"Gender Distribution: Male: {stats['male_count']}, Female: {stats['female_count']}")
    print("Age Groups:")
    for age_group, count in stats['age_groups'].items():
        print(f"  {age_group}: {count}")

# Save detailed data to CSV
if data:
    df = pd.DataFrame(data, columns=['ID', 'Gender', 'Age', 'Total Dwell Time', 'Zone', 'Zone Dwell Time'])
    
    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path)
        combined_data = pd.concat([existing_data, df], ignore_index=True)
        combined_data.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)

# Save zone statistics to a separate CSV
zone_stats_data = []
for zone_name, stats in zone_stats.items():
    zone_stats_data.append({
        'Zone': zone_name,
        'Total Visitors': len(stats['visitors']),
        'Avg Dwell Time': stats['avg_dwell_time'],
        'Male Count': stats['male_count'],
        'Female Count': stats['female_count'],
        'Age Groups': str(dict(stats['age_groups']))
    })

zone_stats_df = pd.DataFrame(zone_stats_data)
zone_stats_df.to_csv('zone_statistics.csv', index=False)
