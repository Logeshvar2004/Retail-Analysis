import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
import time

# Title and description
st.title("Retail Data Analysis")
st.markdown("Upload tracking data in CSV format to analyze customer behavior in retail zones.")

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = False

# Sidebar configuration
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload tracking data (CSV)", type=['csv'])
uploaded_zone = st.sidebar.file_uploader("Upload zone JSON file (optional)", type=['json'])

# Sample data option
use_sample_data = st.sidebar.checkbox("Use sample data instead", False)

# Link to main application
if st.sidebar.button("Go to Video Analysis"):
    st.sidebar.info("Redirecting to Video Analysis tool...")
    # Note: In a real application, you would use st.experimental_set_query_params() or similar
    # for navigation between Streamlit pages. Since this is a separate file demo,
    # we're just showing a placeholder message.
    st.sidebar.success("In a real deployment, this would navigate to the video analysis page.")

# About section in sidebar
st.sidebar.markdown("---")
st.sidebar.header("About this tool")
st.sidebar.markdown("""
This tool analyzes retail tracking data from CSV files:

- **Customer Tracking**: Analyze movement patterns from tracking data
- **Zone Analysis**: Understand dwell time in different retail zones
- **Demographics**: Visualize age and gender distribution
- **Insights**: Get actionable insights about customer behavior
""")

# Function to generate sample data if needed
def generate_sample_data():
    # Generate realistic sample data
    frames = []
    n_people = 50
    total_frames = 1000
    
    zones = ["Zone 1", "Zone 2", "Zone 3", "Zone 4"]
    zone_names = {
        "Zone 1": "Grocery",
        "Zone 2": "Home Appliances",
        "Zone 3": "Stationery",
        "Zone 4": "Cosmetics"
    }
    
    age_groups = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genders = ['Male', 'Female']
    
    for obj_id in range(n_people):
        # Each person appears in a random frame and stays for a random duration
        start_frame = np.random.randint(0, total_frames // 2)
        duration = np.random.randint(50, 300)  # How many frames they appear for
        
        # Assign random age and gender
        age = np.random.choice(age_groups)
        gender = np.random.choice(genders)
        
        # Assign a zone or multiple zones
        n_zones = np.random.randint(1, 4)  # Each person visits 1-3 zones
        person_zones = np.random.choice(zones, n_zones, replace=False)
        
        frames_per_zone = duration // len(person_zones)
        
        current_frame = start_frame
        for zone in person_zones:
            zone_duration = frames_per_zone + np.random.randint(-10, 10)  # Add some variance
            dwell_time = zone_duration / 30  # Assuming 30fps
            
            for _ in range(zone_duration):
                if current_frame < total_frames:
                    frames.append({
                        'Frame': current_frame,
                        'Object ID': obj_id,
                        'Age': age,
                        'Gender': gender,
                        'Dwell Time': dwell_time,
                        'Zone': zone,
                        'Retail Zone': zone_names[zone]
                    })
                    current_frame += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(frames)
    return df

# Function to prepare data for analysis
def prepare_data(df):
    # Check if the dataframe has required columns
    required_columns = ['Frame', 'Object ID', 'Age', 'Gender', 'Dwell Time', 'Zone']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return None
    
    # Add retail zone name if not present
    if 'Retail Zone' not in df.columns:
        # Define retail zone mapping
        zone_names = {
            "Zone 1": "Grocery",
            "Zone 2": "Home Appliances",
            "Zone 3": "Stationery",
            "Zone 4": "Cosmetics"
        }
        
        # Apply mapping - use original zone name if not in the mapping
        df['Retail Zone'] = df['Zone'].map(lambda x: zone_names.get(x, x))
    
    return df

# Function to analyze data and generate stats
def analyze_data(df):
    # Initialize stats dictionary
    stats = {}
    
    # Get unique zones
    zones = df['Zone'].unique()
    
    for zone in zones:
        # Filter data for this zone
        zone_data = df[df['Zone'] == zone]
        
        # Get retail zone name
        retail_zone = zone_data['Retail Zone'].iloc[0] if 'Retail Zone' in df.columns else zone
        
        # Count unique visitors
        visitors = zone_data['Object ID'].nunique()
        
        # Calculate average dwell time
        avg_dwell_time = zone_data.groupby('Object ID')['Dwell Time'].max().mean()
        
        # Count gender distribution
        male_count = zone_data[zone_data['Gender'] == 'Male']['Object ID'].nunique()
        female_count = zone_data[zone_data['Gender'] == 'Female']['Object ID'].nunique()
        
        # Count age groups
        age_groups = zone_data.groupby('Age')['Object ID'].nunique().to_dict()
        
        # Store stats
        stats[zone] = {
            'visitors': visitors,
            'avg_dwell_time': avg_dwell_time,
            'male_count': male_count,
            'female_count': female_count,
            'age_groups': age_groups,
            'retail_name': retail_zone
        }
    
    return stats

# Function to display analytics visualizations
def display_analytics(stats):
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
        
        for zone_name, zone_stats in stats.items():
            retail_name = zone_stats['retail_name']
            visitors_data['Zone'].append(zone_name)
            visitors_data['Retail Section'].append(retail_name)
            visitors_data['Visitors'].append(zone_stats['visitors'])
            visitors_data['Avg Dwell Time (s)'].append(round(zone_stats['avg_dwell_time'], 2))
        
        visitors_df = pd.DataFrame(visitors_data)
        
        # Create visitor count chart
        st.write("Number of Visitors per Retail Section")
        if not visitors_df.empty:
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
        else:
            st.warning("No visitor data available.")
    
    with tab2:
        st.subheader("Gender Distribution by Retail Section")
        
        # Create data for the gender chart
        gender_data = {
            'Retail Section': [],
            'Gender': [],
            'Count': []
        }
        
        for zone_name, zone_stats in stats.items():
            retail_name = zone_stats['retail_name']
            gender_data['Retail Section'].append(retail_name)
            gender_data['Gender'].append('Male')
            gender_data['Count'].append(zone_stats['male_count'])
            
            gender_data['Retail Section'].append(retail_name)
            gender_data['Gender'].append('Female')
            gender_data['Count'].append(zone_stats['female_count'])
        
        gender_df = pd.DataFrame(gender_data)
        
        if not gender_df.empty and gender_df['Count'].sum() > 0:
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
            
            zones_list = list(stats.keys())
            for i, zone_name in enumerate(zones_list):
                zone_stats = stats[zone_name]
                retail_name = zone_stats['retail_name']
                total = zone_stats['male_count'] + zone_stats['female_count']
                
                if total > 0:
                    gender_pie_data = pd.DataFrame({
                        'Gender': ['Male', 'Female'],
                        'Count': [zone_stats['male_count'], zone_stats['female_count']]
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
        else:
            st.warning("No gender data available.")
    
    with tab3:
        st.subheader("Age Distribution by Retail Section")
        
        # Create data for the age chart
        age_data = []
        
        for zone_name, zone_stats in stats.items():
            retail_name = zone_stats['retail_name']
            for age_group, count in zone_stats['age_groups'].items():
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
            for zone_name, zone_stats in stats.items():
                retail_name = zone_stats['retail_name']
                if zone_stats['age_groups']:
                    st.write(f"#### Age Distribution for {retail_name}")
                    
                    age_zone_data = []
                    for age_group, count in zone_stats['age_groups'].items():
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
            st.warning("No age data available.")

# Function to add insights based on the analysis
def provide_insights(stats):
    st.header("Key Insights")
    
    if not stats:
        st.warning("Not enough data to generate insights.")
        return
    
    # Find zone with most visitors
    most_visitors_zone = max(stats.items(), key=lambda x: x[1]['visitors'])
    most_visitors_name = most_visitors_zone[1]['retail_name']
    most_visitors_count = most_visitors_zone[1]['visitors']
    
    # Find zone with longest dwell time
    longest_dwell_zone = max(stats.items(), key=lambda x: x[1]['avg_dwell_time'])
    longest_dwell_name = longest_dwell_zone[1]['retail_name']
    longest_dwell_time = longest_dwell_zone[1]['avg_dwell_time']
    
    # Calculate gender distribution
    total_male = sum(stat['male_count'] for stat in stats.values())
    total_female = sum(stat['female_count'] for stat in stats.values())
    total_visitors = total_male + total_female
    
    # Age distribution
    age_counts = {}
    for stat in stats.values():
        for age, count in stat['age_groups'].items():
            if age in age_counts:
                age_counts[age] += count
            else:
                age_counts[age] = count
    
    # Generate insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Most Popular Section", most_visitors_name, f"{most_visitors_count} visitors")
        
        if total_visitors > 0:
            st.metric("Gender Distribution", f"{round(total_male/total_visitors*100, 1)}% Male", 
                     f"{round(total_female/total_visitors*100, 1)}% Female")
    
    with col2:
        st.metric("Highest Dwell Time", longest_dwell_name, f"{round(longest_dwell_time, 2)} seconds")
        
        if age_counts:
            dominant_age = max(age_counts.items(), key=lambda x: x[1])
            st.metric("Dominant Age Group", dominant_age[0], f"{dominant_age[1]} visitors")
    
    # More detailed insights
    st.subheader("Actionable Insights")
    
    insights = []
    
    # Popularity vs dwell time insight
    if most_visitors_zone[0] != longest_dwell_zone[0]:
        insights.append(f"While {most_visitors_name} attracts the most visitors, customers spend more time in {longest_dwell_name}. Consider cross-merchandising strategies between these sections.")
    else:
        insights.append(f"{most_visitors_name} is both the most visited section and has the longest dwell time, indicating strong customer engagement.")
    
    # Gender-based insight
    if total_visitors > 0:
        if total_male > total_female:
            male_dominant_sections = [stat['retail_name'] for zone, stat in stats.items() 
                                     if stat['male_count'] > stat['female_count']]
            if male_dominant_sections:
                insights.append(f"Male shoppers dominate overall ({round(total_male/total_visitors*100, 1)}%), particularly in {', '.join(male_dominant_sections[:2])}. Consider targeting merchandise and promotions accordingly.")
        elif total_female > total_male:
            female_dominant_sections = [stat['retail_name'] for zone, stat in stats.items() 
                                       if stat['female_count'] > stat['male_count']]
            if female_dominant_sections:
                insights.append(f"Female shoppers dominate overall ({round(total_female/total_visitors*100, 1)}%), particularly in {', '.join(female_dominant_sections[:2])}. Consider targeting merchandise and promotions accordingly.")
    
    # Age-based insight
    if age_counts:
        young_age_groups = ['(0-2)', '(4-6)', '(8-12)', '(15-20)']
        adult_age_groups = ['(25-32)', '(38-43)']
        senior_age_groups = ['(48-53)', '(60-100)']
        
        young_count = sum(age_counts.get(age, 0) for age in young_age_groups)
        adult_count = sum(age_counts.get(age, 0) for age in adult_age_groups)
        senior_count = sum(age_counts.get(age, 0) for age in senior_age_groups)
        
        if young_count > adult_count and young_count > senior_count:
            insights.append("Younger demographics constitute a significant portion of your customers. Consider family-friendly layouts and promotions.")
        elif senior_count > young_count and senior_count > adult_count:
            insights.append("Older demographics constitute a significant portion of your customers. Consider accessibility and targeted product placement.")
    
    # Dwell time insight
    low_dwell_zones = [stat['retail_name'] for zone, stat in stats.items() 
                      if stat['avg_dwell_time'] < 30 and stat['visitors'] > 5]
    if low_dwell_zones:
        insights.append(f"Customers spend little time in {', '.join(low_dwell_zones)}. Consider revising product placement or adding engagement features.")
    
    high_dwell_zones = [stat['retail_name'] for zone, stat in stats.items() 
                       if stat['avg_dwell_time'] > 60 and stat['visitors'] > 5]
    if high_dwell_zones:
        insights.append(f"Customers spend significant time in {', '.join(high_dwell_zones)}. These are prime areas for promotional items and cross-selling opportunities.")
    
    # Display insights
    for i, insight in enumerate(insights):
        st.write(f"{i+1}. {insight}")
    
    st.write("---")
    st.write("Note: These insights are generated based on the provided data and should be used as a starting point for further analysis.")

# Main analysis workflow
st.header("Data Upload & Analysis")

df = None
stats = None

# Check if we should use sample data
if use_sample_data:
    st.info("Using sample data for demonstration")
    df = generate_sample_data()
    
    # Display a preview of the sample data
    st.write("Sample Data Preview:")
    st.dataframe(df.head())
    
    # Set the analyzed flag
    st.session_state.analyzed_data = True
    
    # Analyze data
    stats = analyze_data(df)

# Check if we have an uploaded file
elif uploaded_file is not None:
    try:
        # Load and display the data
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        # Prepare the data
        df = prepare_data(df)
        if df is not None:
            # Set the analyzed flag
            st.session_state.analyzed_data = True
            
            # Analyze data
            stats = analyze_data(df)
            
    except Exception as e:
        st.error(f"Error loading or processing the file: {str(e)}")
else:
    st.info("Please upload a CSV file with tracking data or use the sample data option.")

# Create analyze button
analyze_button = st.button("Analyze Data")

# Display analytics when the button is clicked or if data is already analyzed
if analyze_button or st.session_state.analyzed_data:
    if stats:
        # Display progress indication
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate processing
        for i in range(101):
            progress_bar.progress(i)
            status_text.text(f"Processing data... {i}%")
            time.sleep(0.01)
        
        status_text.text("Analysis complete!")
        
        # Display analytics
        display_analytics(stats)
        
        # Provide insights
        provide_insights(stats)
    else:
        st.warning("No data available for analysis. Please upload a valid CSV file or use sample data.")

# Download sample CSV template
st.markdown("---")
st.header("Download Resources")

# Function to generate a sample CSV template
def generate_csv_template():
    template_data = {
        'Frame': [1, 1, 2, 2, 3, 3],
        'Object ID': [0, 1, 0, 1, 0, 1],
        'Age': ['(25-32)', '(15-20)', '(25-32)', '(15-20)', '(25-32)', '(15-20)'],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Dwell Time': [2.5, 1.8, 2.6, 1.9, 2.7, 2.0],
        'Zone': ['Zone 1', 'Zone 2', 'Zone 1', 'Zone 2', 'Zone 1', 'Zone 2']
    }
    return pd.DataFrame(template_data)

# Function to generate a sample zone JSON
def generate_zone_json():
    zone_data = [
        {
            "name": "Zone 1",
            "points": [[100, 100], [300, 100], [300, 300], [100, 300]]
        },
        {
            "name": "Zone 2",
            "points": [[350, 100], [550, 100], [550, 300], [350, 300]]
        },
        {
            "name": "Zone 3",
            "points": [[100, 350], [300, 350], [300, 550], [100, 550]]
        },
        {
            "name": "Zone 4",
            "points": [[350, 350], [550, 350], [550, 550], [350, 550]]
        }
    ]
    return zone_data

# Create download buttons
col1, col2 = st.columns(2)

with col1:
    csv_template = generate_csv_template()
    csv = csv_template.to_csv(index=False)
    st.download_button(
        label="Download CSV Template",
        data=csv,
        file_name="retail_tracking_template.csv",
        mime="text/csv",
    )

with col2:
    zone_json = generate_zone_json()
    json_str = json.dumps(zone_json, indent=2)
    st.download_button(
        label="Download Zone JSON Template",
        data=json_str,
        file_name="retail_zones_template.json",
        mime="application/json",
    )

st.markdown("""
### CSV Format Description

Your CSV file should contain the following columns:
- **Frame**: Frame number in the video sequence
- **Object ID**: Unique identifier for each tracked person
- **Age**: Age group, e.g. '(0-2)', '(4-6)', '(8-12)', '(15-20)', etc.
- **Gender**: 'Male' or 'Female'
- **Dwell Time**: Time spent in seconds
- **Zone**: Zone identifier, e.g. 'Zone 1', 'Zone 2', etc.

Optional column:
- **Retail Zone**: Descriptive name for the zone (will be auto-mapped if not provided)
""")