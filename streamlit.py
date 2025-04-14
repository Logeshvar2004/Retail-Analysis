import streamlit as st
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
import altair as alt
import matplotlib.pyplot as plt
from pathlib import Path

# Title and description
st.title("Retail Zone Analytics")
st.markdown("This application analyzes customer behavior in different retail zones using computer vision.")

# Initialize session state
if 'playing' not in st.session_state:
    st.session_state.playing = False
    
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
    
if 'zone_stats' not in st.session_state:
    st.session_state.zone_stats = None

# Configuration sidebar
st.sidebar.header("Configuration")

# Default paths as fallback
default_video_path = r"D:\Coding\python\Retail\2.mp4"
default_zone_path = r"D:\Coding\python\Retail\zone2.json"

# Detection parameters
confidence_threshold = st.sidebar.slider("Detection confidence", 0.1, 1.0, 0.5)
distance_threshold = st.sidebar.slider("Tracking distance threshold", 10, 100, 50)
show_labels = st.sidebar.checkbox("Show detailed labels", True)
processing_delay = st.sidebar.slider("Processing delay (ms)", 0, 100, 30) / 1000.0

# About section in sidebar
st.sidebar.markdown("---")  # Add a separator
st.sidebar.header("About this application")
st.sidebar.markdown("""
This application uses computer vision to analyze customer behavior in retail spaces:

- **Person Detection**: Identifies people in video using YOLOv8
- **Zone Analysis**: Tracks people as they move between predefined zones
- **Demographics**: Estimates age and gender of customers
- **Analytics**: Provides statistics about visitors and dwell time
""")

# Function to check if file exists
def file_exists(file_path):
    return os.path.isfile(file_path)

# Save uploaded files to disk
def save_uploaded_file(uploaded_file, save_path):
    if uploaded_file is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return save_path
    return None

# Handle file paths
video_path = None
zone_path = None

video_path = default_video_path if file_exists(default_video_path) else None

# Handle uploaded zone file
zone_path = default_zone_path if file_exists(default_zone_path) else None

# Check if we have the required files
if video_path is None:
    st.error(f"Video file not found: {default_video_path}. Please upload a video file.")
if zone_path is None:
    st.error(f"Zone file not found: {default_zone_path}. Please upload a zone JSON file.")

# Function to load ML models with error handling
@st.cache_resource
def load_models():
    """Load and cache ML models"""
    models = {}
    
    # Load YOLO model
    try:
        models['yolo'] = YOLO('yolov8n.pt')
        st.sidebar.success("✅ YOLO model loaded successfully")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load YOLO model: {str(e)}")
        models['yolo'] = None
    
    # Define model paths
    model_paths = {
        'age_prototxt': r'D:\Coding\python\Retail\Models\age_deploy.prototxt',
        'age_model': r'D:\Coding\python\Retail\Models\age_net.caffemodel',
        'gender_prototxt': r'D:\Coding\python\Retail\Models\gender_deploy.prototxt',
        'gender_model': r'D:\Coding\python\Retail\Models\gender_net.caffemodel'
    }
    
    # Check each model file
    models_available = True
    for name, path in model_paths.items():
        if not file_exists(path):
            st.sidebar.error(f"❌ Missing {name}: {path}")
            models_available = False
    
    # Load age and gender models if files exist
    if models_available:
        try:
            models['age_net'] = cv2.dnn.readNetFromCaffe(
                model_paths['age_prototxt'], 
                model_paths['age_model']
            )
            models['gender_net'] = cv2.dnn.readNetFromCaffe(
                model_paths['gender_prototxt'],
                model_paths['gender_model']
            )
            st.sidebar.success("✅ Age and gender models loaded successfully")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load age/gender models: {str(e)}")
            models['age_net'] = None
            models['gender_net'] = None
    else:
        models['age_net'] = None
        models['gender_net'] = None
    
    return models

# Function to calculate the centroid of a bounding box
def calculate_centroid(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy

# Function to check which zone a point is in
def get_zone(point, zones):
    point_obj = Point(point)
    for zone_name, zone_polygon in zones.items():
        if zone_polygon.contains(point_obj):
            return zone_name
    return None

# Function to create a new Kalman filter
def create_kalman_filter(initial_pos):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([  # State transition matrix
        [1, 0, 1, 0],  # x = x + vx
        [0, 1, 0, 1],  # y = y + vy
        [0, 0, 1, 0],  # vx = vx
        [0, 0, 0, 1]   # vy = vy
    ])
    kf.H = np.array([  # Measurement function
        [1, 0, 0, 0],  # Only x and y are measured
        [0, 1, 0, 0]
    ])
    kf.x[:2] = np.array(initial_pos, dtype=np.float32).reshape(-1, 1)
    kf.R *= 10.0  # Measurement uncertainty
    kf.P *= 10.0  # Initial state uncertainty
    kf.Q *= 0.01  # Process uncertainty
    return kf

# Get retail zone name based on zone number
def get_retail_zone_name(zone_name):
    # Map zone names to retail departments
    retail_zones = {
        "Zone 1": "Grocery",
        "Zone 2": "Home Appliances",
        "Zone 3": "Stationery",
        "Zone 4": "Cosmetics"
    }
    
    # Return mapped name or original name if not found
    if zone_name in retail_zones:
        return retail_zones[zone_name]
    return zone_name

# Function to process video
def process_video(video_path, zone_path, analyze_only=False):
    # If this is analyze-only mode and we have stats, return them
    if analyze_only and st.session_state.zone_stats is not None:
        return st.session_state.zone_stats
        
    if not file_exists(video_path):
        st.error(f"Video file not found: {video_path}")
        return None
        
    if not file_exists(zone_path):
        st.error(f"Zone file not found: {zone_path}")
        return None
    
    # Load models
    models = load_models()
    if models['yolo'] is None:
        st.error("Failed to load YOLO model. Cannot continue.")
        return None
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Define age and gender lists
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    gender_list = ['Male', 'Female']
    
    # Load zone data with error handling
    try:
        with open(zone_path, 'r') as f:
            zone_data = json.load(f)
            
        # Convert JSON zone points to Polygon(s)
        zones = {}
        for zone in zone_data:
            polygon_points = [tuple(point) for point in zone['points']]
            zones[zone['name']] = Polygon(polygon_points)
            
    except Exception as e:
        st.error(f"Failed to load zone data: {str(e)}")
        return None
    
    # Initialize tracking data structures
    object_kalman_filters = {}  # KF for each tracked object
    object_centroids = defaultdict(list)  # Centroid history
    next_object_id = 0  # Counter for new object IDs
    entry_times = {}  # When objects first appeared
    positions = {}  # Current position of each object
    object_zones = {}  # Current zone for each object
    zone_entry_times = {}  # When object entered each zone
    zone_dwell_times = defaultdict(lambda: defaultdict(float))  # Dwell time in each zone
    
    # Zone-specific statistics
    zone_stats = {zone_name: {
        'visitors': set(),
        'avg_dwell_time': 0,
        'male_count': 0,
        'female_count': 0,
        'age_groups': defaultdict(int),
        'male_tracked': set(),
        'female_tracked': set(),
        'age_tracked': {},
        'retail_name': get_retail_zone_name(zone_name)  # Add retail department name
    } for zone_name in zones.keys()}
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Failed to open video: {video_path}")
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video display placeholder
    video_placeholder = st.empty()
    
    frame_count = 0
    
    # Main video processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        frame_count += 1
        progress = int(frame_count / total_frames * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress}%) - Objects tracked: {len(object_kalman_filters)}")
        
        # Run object detection
        results = models['yolo'](frame, conf=confidence_threshold)
        
        # Prepare lists for new detections
        new_centroids = []
        new_boxes = []
        new_zones = []
        
        # Extract person detections
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    label = box.cls.item()
                    if label == 0:  # 'person' class in COCO dataset
                        confidence = box.conf.item()
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        centroid = calculate_centroid((x1, y1, x2, y2))
                        
                        # Check which zone the person is in
                        zone_name = get_zone(centroid, zones)
                        if zone_name:  # Only track people in defined zones
                            new_centroids.append(centroid)
                            new_boxes.append((x1, y1, x2, y2))
                            new_zones.append(zone_name)
        
        # Make predictions for existing objects
        predicted_centroids = {}
        for object_id, kf in object_kalman_filters.items():
            kf.predict()
            predicted_centroids[object_id] = (int(kf.x[0]), int(kf.x[1]))
        
        # Match objects using Hungarian algorithm
        new_ids = []
        if predicted_centroids and new_centroids:
            # Build cost matrix
            D = np.zeros((len(predicted_centroids), len(new_centroids)))
            for i, (object_id, pred_centroid) in enumerate(predicted_centroids.items()):
                for j, new_centroid in enumerate(new_centroids):
                    D[i, j] = np.linalg.norm(np.array(pred_centroid) - np.array(new_centroid))
            
            # Apply Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(D)
            
            used_rows = set()
            used_cols = set()
            
            # Match existing objects to new detections
            for row, col in zip(row_ind, col_ind):
                if D[row, col] < distance_threshold:  # Only if distance is below threshold
                    object_id = list(predicted_centroids.keys())[row]
                    current_centroid = new_centroids[col]
                    current_zone = new_zones[col]
                    
                    # Update Kalman filter
                    object_centroids[object_id].append(current_centroid)
                    object_kalman_filters[object_id].update(np.array(current_centroid).reshape(-1, 1))
                    new_ids.append(object_id)
                    
                    # Handle zone transitions
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
                    elif object_id not in object_zones or (object_id not in zone_entry_times or 
                                                          current_zone not in zone_entry_times[object_id]):
                        if object_id not in zone_entry_times:
                            zone_entry_times[object_id] = {}
                        zone_entry_times[object_id][current_zone] = time.time()
                    
                    # Update current zone
                    object_zones[object_id] = current_zone
                    
                    used_rows.add(row)
                    used_cols.add(col)
            
            # Create new objects for unmatched detections
            for col in set(range(len(new_centroids))).difference(used_cols):
                new_id = next_object_id
                current_centroid = new_centroids[col]
                current_zone = new_zones[col]
                
                # Create new Kalman filter
                object_kalman_filters[new_id] = create_kalman_filter(current_centroid)
                object_centroids[new_id] = [current_centroid]
                new_ids.append(new_id)
                
                # Record entry time and position
                entry_times[new_id] = time.time()
                positions[new_id] = current_centroid
                
                # Initialize zone tracking
                object_zones[new_id] = current_zone
                zone_entry_times[new_id] = {current_zone: time.time()}
                
                next_object_id += 1
        
        # If no existing objects to match with
        elif new_centroids:
            for i in range(len(new_centroids)):
                new_id = next_object_id
                current_centroid = new_centroids[i]
                current_zone = new_zones[i]
                
                # Create new Kalman filter
                object_kalman_filters[new_id] = create_kalman_filter(current_centroid)
                object_centroids[new_id] = [current_centroid]
                new_ids.append(new_id)
                
                # Record entry time and position
                entry_times[new_id] = time.time()
                positions[new_id] = current_centroid
                
                # Initialize zone tracking
                object_zones[new_id] = current_zone
                zone_entry_times[new_id] = {current_zone: time.time()}
                
                next_object_id += 1
        
        # Process each detected person
        for box, object_id, i in zip(new_boxes, new_ids, range(len(new_zones))):
            x1, y1, x2, y2 = box
            current_zone = new_zones[i]
            
            # Add to zone visitors
            zone_stats[current_zone]['visitors'].add(object_id)
            
            # Analyze age and gender if models available
            gender = "Unknown"
            age = "Unknown"
            
            if models['age_net'] is not None and models['gender_net'] is not None:
                # Extract face region (simplified - using upper body region)
                face_height = int((y2 - y1) * 0.4)  # Take upper 40% of body box
                face = frame[y1:y1+face_height, x1:x2]
                
                if face.size > 0 and face.shape[0] > 20 and face.shape[1] > 20:
                    try:
                        # Prepare face for prediction
                        blob = cv2.dnn.blobFromImage(
                            face, 1.0, (227, 227), 
                            (78.4263377603, 87.7689143744, 114.895847746), 
                            swapRB=False
                        )
                        
                        # Predict gender
                        models['gender_net'].setInput(blob)
                        gender_preds = models['gender_net'].forward()
                        gender = gender_list[gender_preds[0].argmax()]
                        
                        # Update gender count in zone stats
                        if gender == 'Male' and object_id not in zone_stats[current_zone]['male_tracked']:
                            zone_stats[current_zone]['male_count'] += 1
                            zone_stats[current_zone]['male_tracked'].add(object_id)
                        elif gender == 'Female' and object_id not in zone_stats[current_zone]['female_tracked']:
                            zone_stats[current_zone]['female_count'] += 1
                            zone_stats[current_zone]['female_tracked'].add(object_id)
                        
                        # Predict age
                        models['age_net'].setInput(blob)
                        age_preds = models['age_net'].forward()
                        age = age_list[age_preds[0].argmax()]
                        
                        # Update age group count
                        if 'age_tracked' not in zone_stats[current_zone]:
                            zone_stats[current_zone]['age_tracked'] = {}
                        if age not in zone_stats[current_zone]['age_tracked']:
                            zone_stats[current_zone]['age_tracked'][age] = set()
                        
                        if object_id not in zone_stats[current_zone]['age_tracked'].get(age, set()):
                            zone_stats[current_zone]['age_groups'][age] += 1
                            zone_stats[current_zone]['age_tracked'][age].add(object_id)
                            
                    except Exception as e:
                        # Silently handle errors in age/gender prediction
                        pass
            
            # Calculate dwell times
            current_dwell = 0
            if object_id in zone_entry_times and current_zone in zone_entry_times[object_id]:
                current_dwell = time.time() - zone_entry_times[object_id][current_zone]
            
            # Total dwell time
            total_dwell = time.time() - entry_times[object_id]
            
            # Draw visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Show detailed labels if enabled
            if show_labels:
                retail_zone_name = get_retail_zone_name(current_zone)
                label = f"ID: {object_id}, {gender}, {age}, Zone: {retail_zone_name}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Simplified label
                cv2.putText(frame, f"ID: {object_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw zone polygons
        for zone_name, zone_polygon in zones.items():
            pts = np.array(zone_polygon.exterior.coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            color = (0, 0, 255)  # Red for zones
            cv2.polylines(frame, [pts], True, color, 2)
            
            # Find position for zone label and use retail department name
            x_avg = int(np.mean(pts[:, 0, 0]))
            y_avg = int(np.mean(pts[:, 0, 1]))
            retail_zone_name = get_retail_zone_name(zone_name)
            cv2.putText(frame, retail_zone_name, (x_avg, y_avg), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add statistics overlay to frame
        stats_y = 30
        for zone_name, stats in zone_stats.items():
            visitor_count = len(stats['visitors'])
            retail_zone_name = stats['retail_name']
            cv2.putText(frame, f"{retail_zone_name}: {visitor_count} visitors", 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            stats_y += 30
        
        # Convert frame for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Store current stats in session state
        st.session_state.zone_stats = zone_stats
        
        # Check if we should stop playback
        if not st.session_state.playing and not analyze_only:
            break
            
        # Delay for display
        time.sleep(processing_delay)
    
    # Cleanup
    cap.release()
    
    # Calculate final statistics
    for zone_name, stats in zone_stats.items():
        visitor_count = len(stats['visitors'])
        if visitor_count > 0:
            total_zone_dwell = 0
            for visitor_id in stats['visitors']:
                if visitor_id in zone_dwell_times and zone_name in zone_dwell_times[visitor_id]:
                    total_zone_dwell += zone_dwell_times[visitor_id][zone_name]
            stats['avg_dwell_time'] = total_zone_dwell / visitor_count if visitor_count > 0 else 0
    
    progress_bar.progress(100)
    status_text.text("Processing complete!")
    
    return zone_stats

# Function to create analytics visualizations
def display_analytics(zone_stats):
    st.header("Retail Zone Analytics Results")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Zone Overview", "Gender Distribution", "Age Distribution"])
    
    with tab1:
        st.subheader("Retail Section Visitor Statistics")
        
        # Create data for the visitors chart
        visitors_data = {
            'Zone': [],
            'Retail Section': [],
            'Visitors': [],
            'Avg Dwell Time (s)': []
        }
        
        for zone_name, stats in zone_stats.items():
            retail_name = stats['retail_name']
            visitors_data['Zone'].append(zone_name)
            visitors_data['Retail Section'].append(retail_name)
            visitors_data['Visitors'].append(len(stats['visitors']))
            visitors_data['Avg Dwell Time (s)'].append(round(stats['avg_dwell_time'], 2))
        
        visitors_df = pd.DataFrame(visitors_data)
        
        # Create visitor count chart
        st.write("Number of Visitors per Retail Section")
        visitors_chart = alt.Chart(visitors_df).mark_bar().encode(
            x=alt.X('Retail Section', sort='-y', title="Retail Section"),
            y='Visitors',
            color='Retail Section',
            tooltip=['Retail Section', 'Visitors', 'Avg Dwell Time (s)']
        ).properties(
            height=400
        )
        st.altair_chart(visitors_chart, use_container_width=True)
        
        # Create dwell time chart
        st.write("Average Dwell Time per Retail Section (seconds)")
        dwell_chart = alt.Chart(visitors_df).mark_bar().encode(
            x=alt.X('Retail Section', sort='-y'),
            y='Avg Dwell Time (s)',
            color='Retail Section',
            tooltip=['Retail Section', 'Visitors', 'Avg Dwell Time (s)']
        ).properties(
            height=400
        )
        st.altair_chart(dwell_chart, use_container_width=True)
        
        # Display zone stats as a table
        st.write("Retail Section Statistics")
        display_df = visitors_df[['Retail Section', 'Visitors', 'Avg Dwell Time (s)']]
        st.dataframe(display_df)
    
    with tab2:
        st.subheader("Gender Distribution by Retail Section")
        
        # Create data for the gender chart
        gender_data = {
            'Retail Section': [],
            'Gender': [],
            'Count': []
        }
        
        for zone_name, stats in zone_stats.items():
            retail_name = stats['retail_name']
            gender_data['Retail Section'].append(retail_name)
            gender_data['Gender'].append('Male')
            gender_data['Count'].append(stats['male_count'])
            
            gender_data['Retail Section'].append(retail_name)
            gender_data['Gender'].append('Female')
            gender_data['Count'].append(stats['female_count'])
        
        gender_df = pd.DataFrame(gender_data)
        
        # Create gender distribution chart
        gender_chart = alt.Chart(gender_df).mark_bar().encode(
            x='Retail Section',
            y='Count',
            color='Gender',
            tooltip=['Retail Section', 'Gender', 'Count']
        ).properties(
            height=400
        )
        st.altair_chart(gender_chart, use_container_width=True)
        
        # Create gender pie charts for each zone
        col1, col2 = st.columns(2)
        
        zones_list = list(zone_stats.keys())
        for i, zone_name in enumerate(zones_list):
            stats = zone_stats[zone_name]
            retail_name = stats['retail_name']
            total = stats['male_count'] + stats['female_count']
            
            if total > 0:
                gender_pie_data = pd.DataFrame({
                    'Gender': ['Male', 'Female'],
                    'Count': [stats['male_count'], stats['female_count']]
                })
                
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(gender_pie_data['Count'], labels=gender_pie_data['Gender'], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                plt.title(f"Gender Distribution - {retail_name}")
                
                if i % 2 == 0:
                    col1.pyplot(fig)
                else:
                    col2.pyplot(fig)
            else:
                if i % 2 == 0:
                    col1.write(f"No gender data for {retail_name}")
                else:
                    col2.write(f"No gender data for {retail_name}")
    
    with tab3:
        st.subheader("Age Distribution by Retail Section")
        
        # Create data for the age chart
        age_data = []
        
        for zone_name, stats in zone_stats.items():
            retail_name = stats['retail_name']
            for age_group, count in stats['age_groups'].items():
                age_data.append({
                    'Retail Section': retail_name,
                    'Age Group': age_group,
                    'Count': count
                })
        
        if age_data:
            age_df = pd.DataFrame(age_data)
            
            # Create age distribution chart
            age_chart = alt.Chart(age_df).mark_bar().encode(
                x='Retail Section',
                y='Count',
                color='Age Group',
                tooltip=['Retail Section', 'Age Group', 'Count']
            ).properties(
                height=400
            )
            st.altair_chart(age_chart, use_container_width=True)
            
            # Create detailed age distribution for each zone
            for zone_name, stats in zone_stats.items():
                retail_name = stats['retail_name']
                if stats['age_groups']:
                    st.write(f"#### Age Distribution for {retail_name}")
                    
                    age_zone_data = []
                    for age_group, count in stats['age_groups'].items():
                        age_zone_data.append({
                            'Age Group': age_group,
                            'Count': count
                        })
                    
                    age_zone_df = pd.DataFrame(age_zone_data)
                    
                    # Create bar chart for age distribution
                    age_zone_chart = alt.Chart(age_zone_df).mark_bar().encode(
                        x='Age Group',
                        y='Count',
                        color='Age Group',
                        tooltip=['Age Group', 'Count']
                    ).properties(
                        height=300
                    )
                    st.altair_chart(age_zone_chart, use_container_width=True)
                else:
                    st.write(f"No age data available for {retail_name}")
        else:
            st.write("No age data available.")

# Create buttons for control
col1, col2 = st.columns(2)

# Play/Pause button
if col1.button("Play/Pause Video"):
    st.session_state.playing = not st.session_state.playing

# Analyze button
if col2.button("Show Analytics"):
    st.session_state.analyzed = True
    # Run the analysis in non-playing mode (just analyze once)
    zone_stats = process_video(video_path, zone_path, analyze_only=True)
    if zone_stats:
        display_analytics(zone_stats)
    else:
        st.error("Analysis failed. Please check the error messages above.")

# Run the video processor if playing
if video_path and zone_path and st.session_state.playing:
    zone_stats = process_video(video_path, zone_path)
    st.session_state.playing = False  # Stop after processing
elif st.session_state.analyzed and st.session_state.zone_stats:
    # Display analytics if in analyzed state
    display_analytics(st.session_state.zone_stats)
else:
    # Show instructions when not playing
    if not st.session_state.analyzed:
        st.info("Click 'Play/Pause Video' to start processing or 'Show Analytics' to view results.")