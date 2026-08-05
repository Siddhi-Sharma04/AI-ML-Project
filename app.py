import os
import cv2
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configure Streamlit page options first
st.set_page_config(
    page_title="AI Traffic Control & E-Challan System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom core modules
from core_cv_pipeline.database import (
    init_db, clear_db, get_metrics, get_all_challans, 
    get_all_vehicle_logs, update_challan_status
)
from core_cv_pipeline.main import load_models, process_video

# Create weights and output dirs if not exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "weights"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "generated_challans"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "assets", "detected_plates"), exist_ok=True)

# Define file paths
tracker_path = os.path.join(BASE_DIR, "weights", "yolov8n.pt")
plate_path = os.path.join(BASE_DIR, "weights", "license_plate_detector.pt")
helmet_path = os.path.join(BASE_DIR, "weights", "helmet_detector.pt")
zones_path = os.path.join(BASE_DIR, "core_cv_pipeline", "config", "zones.json")

# Initialize database
init_db()

# Caching model loading to avoid redundant loads across streamlit runs
@st.cache_resource
def get_cached_models():
    return load_models(
        tracker_weights=tracker_path,
        plate_weights=plate_path,
        helmet_weights=helmet_path,
        zones_config=zones_path
    )

try:
    models = get_cached_models()
except Exception as e:
    st.error(f"Error loading AI models: {e}. Ensure weights files are in the 'weights/' directory.")
    models = None

# Custom CSS for Premium Glassmorphism Dark Mode Styling
st.markdown("""
<style>
    .reportview-container {
        background-color: #0B0F19;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #FFFFFF;
        font-weight: 700;
    }
    .stButton>button {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF2B6B 100%);
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.5);
    }
    .card-container {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #3B82F6;
        line-height: 1;
        margin-top: 0.5rem;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("<h2 style='text-align: center; color: #FF4B4B;'>🚦 Traffic AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 0.8rem; color: #9CA3AF;'>Automated Violation Detection Dashboard</p>", unsafe_allow_html=True)
st.sidebar.write("---")

# Navigation options
page = st.sidebar.radio(
    "Navigation", 
    ["Dashboard", "Analytics", "Challan Log", "Settings", "About Project"],
    index=0
)

# Active database and backend status ping indicators
st.sidebar.write("---")
st.sidebar.markdown("### System Status")
st.sidebar.markdown("🟢 **Database & SQL engine**: Connected")
if models:
    st.sidebar.markdown("🟢 **YOLO Detection engine**: Loaded")
else:
    st.sidebar.markdown("🔴 **AI Models**: Missing weights")

# ----------------- PAGES -----------------

# Page: Dashboard
if page == "Dashboard":
    st.markdown("## Real-Time Violation & Detection Dashboard")
    st.write("Monitor camera traffic streams, verify license plates, and generate automated e-challans.")

    # Live Metrics section
    metrics = get_metrics()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="card-container">
            <div class="metric-label">Vehicles Tracked</div>
            <div class="metric-value">{metrics["total_vehicles"]}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card-container">
            <div class="metric-label">Violations Detected</div>
            <div class="metric-value" style="color: #EF4444;">{metrics["total_violations"]}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="card-container">
            <div class="metric-label">Total Fines Issued</div>
            <div class="metric-value" style="color: #F59E0B;">₹{metrics["total_fines"]:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="card-container">
            <div class="metric-label">Active Monitors</div>
            <div class="metric-value" style="color: #10B981;">1 / 1 Online</div>
        </div>
        """, unsafe_allow_html=True)

    # Core Video feed and Control Row
    feed_col, telemetry_col = st.columns([2, 1])

    with feed_col:
        st.markdown("### Video Analysis Feed")
        
        # Check files inside videos directory
        videos_dir = os.path.join(BASE_DIR, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        video_files = [f for f in os.listdir(videos_dir) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
        
        # User upload or dropdown options
        uploaded_file = st.file_uploader("Upload video file (MP4, AVI)", type=["mp4", "avi", "mov", "mkv"])
        
        video_source_path = None
        if uploaded_file:
            temp_path = os.path.join(videos_dir, f"temp_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_source_path = temp_path
        elif video_files:
            selected_video = st.selectbox("Or choose a pre-loaded sample video file", video_files)
            video_source_path = os.path.join(videos_dir, selected_video)
        else:
            st.warning("Please upload a video file or place one inside the 'videos/' directory.")

        # Performance Tuning for Demonstration Speed (FPS Control)
        perf_mode = st.select_slider(
            "Inference Speed Optimization (Frame Skip)",
            options=["High Accuracy (Slow)", "Balanced (Recommended)", "Max Speed (Turbo)"],
            value="Balanced (Recommended)",
            help="Balanced skips redundant frames to reach 6-8 FPS. Turbo skips more to reach 12-15 FPS. High Accuracy processes every frame."
        )
        
        frame_skip_map = {
            "High Accuracy (Slow)": 1,
            "Balanced (Recommended)": 3,
            "Max Speed (Turbo)": 5
        }
        forced_skip = frame_skip_map[perf_mode]

        start_btn = st.button("Start Detection")
        
        # Canvas frame placeholder
        frame_placeholder = st.empty()
        
        if start_btn and video_source_path:
            if not models:
                st.error("Cannot start detection because model weights are not loaded.")
            else:
                st.info("Starting Computer Vision pipeline loops...")
                progress_bar = st.progress(0)
                
                # Run the CV generator loop and stream frames to Streamlit
                for update in process_video(video_source_path, zones_path, models, frame_skip_forced=forced_skip):
                    # Display frame (BGR to RGB conversion for Streamlit rendering)
                    frame_rgb = cv2.cvtColor(update["frame"], cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, use_container_width=True)
                    
                    # Update progress bar
                    progress_bar.progress(int(update["progress_pct"]) / 100.0)
                    
                    # Render live stats
                    telemetry_col.markdown(f"""
                    **Status:** Processing  
                    **Time Elapsed:** {update["elapsed"]:.1f}s  
                    **Processing Speed:** {update["fps"]:.1f} FPS  
                    **Tracked in Frame:** {update["vehicles_in_frame"]} vehicles  
                    **Density Status:** {update["traffic_status"]}  
                    """)
                    time.sleep(0.01) # Small pause for yield refresh
                
                progress_bar.empty()
                st.success("Analysis complete!")
                st.rerun()

    with telemetry_col:
        st.markdown("### Telemetry Status")
        st.write("Start a video analysis feed to see real-time computer vision metrics.")

    # Live Database logs
    st.write("---")
    st.markdown("### Latest Registered Challans & Logs")
    challans = get_all_challans(limit=10)
    
    if challans:
        df = pd.DataFrame(challans)
        # Drop columns not suitable for main view
        display_df = df[["challan_number", "timestamp", "owner_name", "license_plate", "violation_types", "fine_amount", "status"]].copy()
        # Clean formatting
        display_df["violation_types"] = display_df["violation_types"].apply(lambda v: ", ".join(v) if isinstance(v, list) else v)
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No violations detected yet in this session.")

# Page: Analytics
elif page == "Analytics":
    st.markdown("## Interactive Traffic Analytics")
    st.write("View compiled statistical charts generated from database records.")
    
    challans = get_all_challans(limit=1000)
    vehicle_logs = get_all_vehicle_logs(limit=1000)

    if not challans:
        st.warning("Please run a video analysis session to populate database tables with analytics records first.")
    else:
        # Load data frames
        df_c = pd.DataFrame(challans)
        df_v = pd.DataFrame(vehicle_logs)

        tab1, tab2 = st.tabs(["Violation Analytics", "Traffic Flow Dynamics"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Donut Chart for Violation Distribution
                all_viols = []
                for v_list in df_c["violation_types"]:
                    if isinstance(v_list, list):
                        all_viols.extend(v_list)
                    else:
                        all_viols.append(str(v_list))
                
                v_counts = pd.Series(all_viols).value_counts().reset_index()
                v_counts.columns = ["Violation", "Count"]
                
                fig = px.pie(v_counts, values="Count", names="Violation", title="Violation Types Distribution", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Fines by Vehicle Type
                fig2 = px.bar(df_c, x="license_plate", y="fine_amount", title="Fine Distribution by Vehicle License Plate", labels={"license_plate": "Vehicle Plate", "fine_amount": "Fine Amount (INR)"}, color="fine_amount")
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                # Speed distribution histogram
                if not df_v.empty:
                    fig3 = px.histogram(df_v, x="detected_speed", title="Detected Speed Distribution Profile", labels={"detected_speed": "Speed (km/h)"}, color_discrete_sequence=["#3B82F6"], nbins=20)
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("No speed metrics logs collected yet.")
            with col2:
                # Distribution of tracked vehicle classes
                if not df_v.empty:
                    type_counts = df_v["vehicle_type"].value_counts().reset_index()
                    type_counts.columns = ["Vehicle Type", "Count"]
                    fig4 = px.pie(type_counts, values="Count", names="Vehicle Type", title="Tracked Vehicle Classes Profile", hole=0.3)
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("No vehicle logs collected yet.")

# Page: Challan Log
elif page == "Challan Log":
    st.markdown("## Challan Registry & Official Invoices")
    st.write("Search, view details, and download generated PDF e-challans.")

    challans = get_all_challans(limit=1000)
    
    if not challans:
        st.info("No e-challan tickets logged in database yet.")
    else:
        df = pd.DataFrame(challans)
        
        # Search & Filter controls
        col1, col2 = st.columns(2)
        with col1:
            search_plate = st.text_input("Filter by License Plate:").upper().strip()
        with col2:
            status_filter = st.selectbox("Filter by Status:", ["All", "Pending", "Paid"])

        # Filter operations
        filtered_df = df.copy()
        if search_plate:
            filtered_df = filtered_df[filtered_df["license_plate"].str.contains(search_plate)]
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df["status"] == status_filter]

        # Display results
        for idx, row in filtered_df.iterrows():
            with st.expander(f"Challan #{row['challan_number']} | Plate: {row['license_plate']} | Fine: ₹{row['fine_amount']} | Status: {row['status']}"):
                col_info, col_img = st.columns([2, 1])
                
                with col_info:
                    st.markdown(f"**Date Issued:** {row['timestamp'].strftime('%d-%b-%Y %I:%M %p')}")
                    st.markdown(f"**Owner Name:** {row['owner_name']}")
                    st.markdown(f"**Owner Phone:** {row['owner_phone']}")
                    st.markdown(f"**Violation Type(s):** {', '.join(row['violation_types']) if isinstance(row['violation_types'], list) else row['violation_types']}")
                    st.markdown(f"**Fine Amount:** ₹{row['fine_amount']}")
                    
                    # Update status paid/pending toggle
                    if row["status"] == "Pending":
                        if st.button("Mark as Paid", key=f"pay_{row['challan_number']}"):
                            update_challan_status(row["challan_number"], "Paid")
                            st.success(f"Challan {row['challan_number']} successfully marked as Paid!")
                            st.rerun()
                            
                    # Download button for E-Challan PDF
                    pdf_path = row["pdf_path"]
                    if os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                        st.download_button(
                            label="Download Challan PDF",
                            data=pdf_bytes,
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf",
                            key=f"dl_{row['challan_number']}"
                        )
                    else:
                        st.warning("PDF Challan document not found on disk.")
                        
                with col_img:
                    evidence_img_path = row["evidence_image_path"]
                    if os.path.exists(evidence_img_path):
                        st.image(evidence_img_path, caption="Evidence Snapshot", use_container_width=True)
                    else:
                        st.warning("Evidence image snapshot not found.")

# Page: Settings
elif page == "Settings":
    st.markdown("## Application Settings & Configurations")
    
    st.write("### Database Cleanup")
    if st.button("Clear SQLite Database"):
        clear_db()
        st.success("Database table records dropped and initialized clean!")
        st.rerun()
        
    st.write("---")
    st.write("### Detection Parameters")
    st.slider("Overspeeding Limit (km/h)", min_value=20, max_value=120, value=40, step=5)
    st.slider("OCR Confidence Threshold", min_value=0.1, max_value=0.9, value=0.15, step=0.05)

# Page: About Project
elif page == "About Project":
    st.markdown("## About AI Traffic Intelligence Project")
    st.write("""
    This project is an **AI-powered Traffic Control & E-Challan Enforcement System** developed as a college major project.
    
    ### Key Features
    - **Speed Traps**: Automatically measures vehicle speed between virtual boundary lines and detects overspeeding violations.
    - **Direction Classification**: Detects wrong-way driving in traffic lanes.
    - **Helmet & Occupant Counts**: Detects motorcycle riders, identifies if they have helmet protection, and triggers alerts for triple-riding.
    - **Automatic Challan Generation**: Renders annotated evidence snapshots and compiles official ReportLab PDF challan invoices.
    
    ### Technologies Used
    - **Computer Vision**: YOLOv8, ByteTrack, OpenCV
    - **OCR**: EasyOCR + CLAHE preprocessing
    - **Database**: SQLite + SQLAlchemy
    - **Dashboard UI**: Streamlit + Plotly
    """)
