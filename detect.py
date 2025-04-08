
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import time
import io
import plotly.graph_objects as go
import requests
import json
import pandas as pd
from datetime import datetime

# Set Streamlit page configuration
st.set_page_config(
    page_title="AI Vision Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load Lottie animations
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Custom CSS with Bootstrap components and enhanced styling
def load_custom_css():
    st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        * { font-family: 'Poppins', sans-serif; }
        .main { background-color: #f3f4f6; padding: 2rem; }
        .stApp { max-width: 1400px; margin: 0 auto; }
        div[data-testid="stVerticalBlock"] { gap: 0 !important; }
        .card { border-radius: 15px; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08); margin-bottom: 25px; 
                background-color: white; padding: 25px; transition: all 0.3s ease; }
        .card:hover { transform: translateY(-5px); box-shadow: 0 12px 30px rgba(0, 0, 0, 0.12); }
        .header-container { background: linear-gradient(135deg, #6e8efb 0%, #a777e3 100%); padding: 3rem 2rem; 
                          border-radius: 15px; margin-bottom: 2rem; color: white; text-align: center; 
                          box-shadow: 0 10px 30px rgba(110, 142, 251, 0.3); position: relative; overflow: hidden; }
        .header-container::before { content: ""; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; 
                                   background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 60%); 
                                   pointer-events: none; }
        .header-container h1 { font-weight: 700; margin-bottom: 1rem; font-size: 2.8rem; letter-spacing: -0.5px; }
        .header-container p { font-size: 1.2rem; max-width: 700px; margin: 0 auto

; opacity: 0.9; }
        .btn-primary { background: linear-gradient(135deg, #6e8efb 0%, #a777e3 100%); border: none; color: white; 
                      padding: 12px 28px; border-radius: 50px; font-weight: 600; margin: 20px 0; 
                      transition: all 0.3s ease; box-shadow: 0 5px 15px rgba(110, 142, 251, 0.4); }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(110, 142, 251, 0.6); 
                           background: linear-gradient(135deg, #5d7de8 0%, #9966d5 100%); }
        .select-container { margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 12px; }
        .select-container h4 { color: #333; margin-bottom: 20px; font-weight: 600; }
        .upload-container { border: 2px dashed #c7d2fe; border-radius: 12px; padding: 40px 20px; 
                          text-align: center; margin: 20px 0; background-color: #f5f7ff; 
                          transition: all 0.3s ease; cursor: pointer; }
        .upload-container:hover { border-color: #6e8efb; background-color: #f0f4ff; }
        .upload-container i { font-size: 3rem; color: #6e8efb; margin-bottom: 1rem; }
        .result-section { margin-top: 30px; border-top: 1px solid #e9ecef; padding-top: 30px; }
        .stat-card { background: white; border-radius: 10px; padding: 20px; 
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05); margin-bottom: 15px; text-align: center; }
        .stat-card h4 { color: #6c757d; font-size: 0.9rem; margin-bottom: 10px; }
        .stat-card .value { font-size: 2rem; font-weight: 700; color: #333; }
        .webcam-container { position: relative; border-radius: 15px; overflow: hidden; 
                          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15); }
        .excel-btn { background-color: #107C41; color: white; border: none; padding: 10px 20px; 
                   border-radius: 5px; cursor: pointer; transition: all 0.3s ease; }
        .excel-btn:hover { background-color: #0D6535; }
        .chart-container { margin-top: 20px; border-radius: 10px; overflow: hidden; background: white; 
                         padding: 15px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05); }
    </style>
    """, unsafe_allow_html=True)

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolo11l.pt")

# Extended COCO dataset classes with additional items
COCO_CLASSES = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4,
    "bus": 5, "train": 6, "truck": 7, "boat": 8, "traffic light": 9,
    "fire hydrant": 10, "stop sign": 11, "parking meter": 12, "bench": 13,
    "bird": 14, "cat": 15, "dog": 16, "horse": 17, "sheep": 18, "cow": 19,
    "elephant": 20, "bear": 21, "zebra": 22, "giraffe": 23, "backpack": 24,
    "umbrella": 25, "handbag": 26, "tie": 27, "suitcase": 28, "frisbee": 29,
    "skis": 30, "snowboard": 31, "sports ball": 32, "kite": 33, "baseball bat": 34,
    "baseball glove": 35, "skateboard": 36, "surfboard": 37, "tennis racket": 38,
    "bottle": 39, "wine glass": 40, "cup": 41, "fork": 42, "knife": 43,
    "spoon": 44, "bowl": 45, "banana": 46, "apple": 47, "sandwich": 48,
    "orange": 49, "broccoli": 50, "carrot": 51, "hot dog": 52, "pizza": 53,
    "donut": 0, "cake": 55, "chair": 56, "couch": 57, "potted plant": 58,
    "bed": 59, "dining table": 60, "toilet": 61, "tv": 62, "laptop": 63,
    "mouse": 64, "remote": 65, "keyboard": 66, "cell phone": 67, "microwave": 68,
    "oven": 69, "toaster": 70, "sink": 71, "refrigerator": 72, "book": 73,
    "clock": 74, "vase": 75, "scissors": 76, "teddy bear": 77, "hair drier": 78,
    "toothbrush": 79, "sunglasses": 80, "tree": 81
}

# Color palette for object detection
COLORS = {
    "cool": [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120)],
    "warm": [(214, 39, 40), (255, 152, 150), (148, 103, 189), (197, 176, 213)],
    "green": [(44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150)],
    "vibrant": [(227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199)]
}

def draw_3d_box(img, x1, y1, x2, y2, color_scheme="cool", depth_factor=0.3, label=None, confidence=None):
    h, w = img.shape[:2]
    depth = int((x2 - x1) * depth_factor)
    
    color_idx = hash(label) % len(COLORS[color_scheme]) if label else 0
    color = COLORS[color_scheme][color_idx]
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    x1_back = x1 + depth
    y1_back = y1 + depth
    x2_back = min(x2 + depth, w-1)
    y2_back = min(y2 + depth, h-1)
    
    alpha = 0.7
    overlay = img.copy()
    cv2.rectangle(overlay, (x1_back, y1_back), (x2_back, y2_back), color, 2)
    cv2.line(overlay, (x1, y1), (x1_back, y1_back), color, 2)
    cv2.line(overlay, (x2, y1), (x2_back, y1_back), color, 2)
    cv2.line(overlay, (x1, y2), (x1_back, y2_back), color, 2)
    cv2.line(overlay, (x2, y2), (x2_back, y2_back), color, 2)
    
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    if label and confidence:
        label_text = f"{label} ({confidence:.2f})"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), color, -1)
        cv2.putText(img, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

def detect_objects(image, target_class, model, conf_threshold=0.5, color_scheme="cool", draw_all_objects=False):
    img = np.array(image)
    original_img = img.copy()
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    results = model(img)
    detected_objects = 0
    highest_confidence = 0
    all_detected_classes = {}
    detection_details = []
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            class_name = next((name for name, id_value in COCO_CLASSES.items() if id_value == class_id), None)
            
            if class_name:
                if class_name not in all_detected_classes:
                    all_detected_classes[class_name] = {"count": 0, "confidence": 0}
                all_detected_classes[class_name]["count"] += 1
                all_detected_classes[class_name]["confidence"] = max(all_detected_classes[class_name]["confidence"], confidence)
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detection_details.append({
                    "class": class_name,
                    "confidence": confidence,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "position_x": (x1 + x2) / 2,
                    "position_y": (y1 + y2) / 2,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            
            if (class_id == COCO_CLASSES.get(target_class.lower(), -1) and confidence > conf_threshold) or \
               (draw_all_objects and confidence > conf_threshold):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                img = draw_3d_box(img, x1, y1, x2, y2, color_scheme, label=class_name, confidence=confidence)
                
                if class_id == COCO_CLASSES.get(target_class.lower(), -1):
                    detected_objects += 1
                    highest_confidence = max(highest_confidence, confidence)
    
    if detected_objects == 0 and not draw_all_objects:
        message = f"No {target_class} detected"
        cv2.putText(img, message, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img, detected_objects, highest_confidence, all_detected_classes, original_img, detection_details

def get_binary_file_downloader_html(bin_data, file_label='File', file_name="download.png", icon="download", btn_class="btn-primary"):
    b64 = base64.b64encode(bin_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}" class="btn {btn_class}">' +\
           f'<i class="fas fa-{icon}"></i> Download {file_label}</a>'
    return href

def image_to_bytes(image):
    img_pil = Image.fromarray(image)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

def create_class_distribution_chart(detected_classes):
    if not detected_classes:
        return None
    
    labels = list(detected_classes.keys())
    counts = [data["count"] for data in detected_classes.values()]
    confidences = [data["confidence"] for data in detected_classes.values()]
    
    sorted_indices = np.argsort(counts)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices][:10]
    sorted_counts = [counts[i] for i in sorted_indices][:10]
    sorted_confidences = [confidences[i] for i in sorted_indices][:10]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sorted_labels, y=sorted_counts, name='Count', 
                        marker_color='rgba(110, 142, 251, 0.8)', text=sorted_counts, textposition='auto'))
    fig.add_trace(go.Scatter(x=sorted_labels, y=sorted_confidences, name='Confidence', yaxis='y2',
                           line=dict(color='rgba(167, 119, 227, 0.8)', width=3), mode='lines+markers', 
                           marker=dict(size=8)))
    
    fig.update_layout(
        title='Object Detection Results',
        xaxis=dict(title='Detected Classes'),
        yaxis=dict(title='Count', side='left'),
        yaxis2=dict(title='Confidence', side='right', overlaying='y', range=[0, 1], tickformat='.0%'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=80),
        plot_bgcolor='white'
    )
    return fig

def create_pie_chart(detected_classes):
    if not detected_classes:
        return None
    
    labels = list(detected_classes.keys())
    counts = [data["count"] for data in detected_classes.values()]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=counts,
        hole=.3,
        textinfo='percent+label',
        marker=dict(colors=['rgba(110, 142, 251, 0.8)', 'rgba(167, 119, 227, 0.8)', 
                           'rgba(214, 39, 40, 0.8)', 'rgba(44, 160, 44, 0.8)',
                           'rgba(255, 127, 14, 0.8)', 'rgba(31, 119, 180, 0.8)'])
    )])
    
    fig.update_layout(
        title='Object Class Distribution',
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def generate_gallery_images():
    return [
        {"title": "Car Detection", "image": "https://i.ytimg.com/vi/POqBiiLaslk/sddefault.jpg", "description": "YOLO v11 detecting cars in traffic"},
        {"title": "Pet Detection", "image": "https://ai.google.dev/static/mediapipe/images/solutions/examples/object_detector.png", "description": "Finding cats and dogs"},
        {"title": "Person Detection", "image": "https://www.exposit.com/wp-content/uploads/2022/12/how-to-raise-fifa-champion-main-1024x386.jpg", "description": "People counting application"}
    ]

def generate_excel(detection_details):
    if not detection_details:
        return None
    
    df = pd.DataFrame(detection_details)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Detection Results', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Detection Results']
        
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4B0082',
            'font_color': 'white',
            'border': 1
        })
        
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).apply(len).max(), len(str(col)) + 2)
            worksheet.set_column(i, i, max_len)
    
    return output.getvalue()

def main():
    load_custom_css()
    model = load_model()
    
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    if 'settings' not in st.session_state:
        st.session_state.settings = {'color_scheme': 'cool', 'draw_all_objects': False, 'depth_factor': 0.3}
    
    if 'last_detection_details' not in st.session_state:
        st.session_state.last_detection_details = None
    
    with st.sidebar:
        st.image("https://www.repeato.app/wp-content/uploads/2024/06/AI-computer-vision-automation.jpg", width=150)
        st.title("AI Vision Lab")
        st.subheader("Global Settings")
        st.session_state.settings['color_scheme'] = st.selectbox(
            "Color Scheme", 
            options=list(COLORS.keys()),
            index=list(COLORS.keys()).index(st.session_state.settings['color_scheme'])
        )
        st.session_state.settings['draw_all_objects'] = st.checkbox(
            "Detect All Objects", 
            value=st.session_state.settings['draw_all_objects']
        )
        st.session_state.settings['depth_factor'] = st.slider(
            "3D Effect Depth", 
            min_value=0.1, 
            max_value=0.5, 
            value=st.session_state.settings['depth_factor'], 
            step=0.05  # Fixed syntax error: 'step woj=0.05' to 'step=0.05'
        )
        
        st.markdown("---")
        st.markdown("#### Recent Detections")
        if st.session_state.detection_history:
            for item in st.session_state.detection_history[-3:]:
                st.markdown(f"**{item['target']}**: {item['count']} objects found")
            if len(st.session_state.detection_history) > 3:
                st.markdown(f"... and {len(st.session_state.detection_history) - 3} more")
        else:
            st.markdown("No detections yet")
    
    st.markdown("""
    <div class="header-container">
        <h1><i class="fas fa-eye"></i> AI Vision Lab</h1>
        <p>Experience next-generation object detection with stunning 3D visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📸 Image Detection", "📹 Live Detection", "📊 Analytics", "🖼️ Gallery", "ℹ️ About"])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            <div class="upload-container">
                <i class="fas fa-cloud-upload-alt"></i>
                <h4>Upload Your Image</h4>
                <p>Drop an image here or click to upload</p>
            </div>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                with st.expander("Image Preview", expanded=True):
                    st.markdown(f"**File name**: {uploaded_file.name}")
                    st.markdown(f"**Image size**: {image.width} x {image.height} pixels")
                    st.markdown(f"**File type**: {uploaded_file.type}")
        
        with col2:
            st.markdown('<div class="select-container">', unsafe_allow_html=True)
            st.markdown("<h4>Detection Settings</h4>", unsafe_allow_html=True)
            target_class = st.selectbox("Select object to detect", options=list(COCO_CLASSES.keys()), index=0)
            conf_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
            detect_button = st.button("Detect Objects", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file and detect_button:
            with st.spinner("Detecting objects..."):
                result_img, count, confidence, detected_classes, _, detection_details = detect_objects(
                    image, 
                    target_class, 
                    model, 
                    conf_threshold, 
                    st.session_state.settings['color_scheme'],
                    st.session_state.settings['draw_all_objects']
                )
                
                st.session_state.last_detection_details = detection_details
                st.session_state.detection_history.append({'target': target_class, 'count': count, 'timestamp': time.time()})
                
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                st.subheader("Detection Results")
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    st.image(result_img, caption="Processed Image", use_container_width=True)
                    st.markdown(get_binary_file_downloader_html(
                        image_to_bytes(result_img), 
                        "Processed Image", 
                        f"detected_{uploaded_file.name}"
                    ), unsafe_allow_html=True)
                    
                    if detection_details:
                        excel_data = generate_excel(detection_details)
                        st.markdown(get_binary_file_downloader_html(
                            excel_data, 
                            "Excel Report", 
                            f"detection_report_{int(time.time())}.xlsx",
                            icon="file-excel",
                            btn_class="excel-btn"
                        ), unsafe_allow_html=True)
                
                with col_result2:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.markdown(f"**Target ({target_class}) Found**: <span class='value'>{count}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Highest Confidence**: <span class='value'>{confidence:.2%}</span>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if detected_classes:
                        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                        st.markdown("**All Detected Objects Count**", unsafe_allow_html=True)
                        for class_name, data in detected_classes.items():
                            st.markdown(f"{class_name}: <span class='value'>{data['count']}</span>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        chart = create_class_distribution_chart(detected_classes)
                        st.plotly_chart(chart, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        pie_chart = create_pie_chart(detected_classes)
                        st.plotly_chart(pie_chart, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Live Webcam Detection")
        
        col_web1, col_web2 = st.columns([3, 2])
        
        with col_web1:
            st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
            camera = st.camera_input("Webcam Feed", key="webcam_feed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_web2:
            st.markdown('<div class="select-container">', unsafe_allow_html=True)
            webcam_target = st.selectbox("Select object to detect", options=list(COCO_CLASSES.keys()), index=0, key="webcam_target")
            webcam_conf = st.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05, key="webcam_conf")
            
            if st.button("Clear Camera", key="clear_camera"):
                st.session_state.pop("webcam_feed", None)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        if camera:
            camera_image = Image.open(camera)
            with st.spinner("Processing webcam image..."):
                result_img, count, confidence, detected_classes, original_img, detection_details = detect_objects(
                    camera_image, 
                    webcam_target, 
                    model, 
                    webcam_conf,
                    st.session_state.settings['color_scheme'],
                    st.session_state.settings['draw_all_objects']
                )
                
                st.session_state.last_detection_details = detection_details
                st.session_state.detection_history.append({
                    'target': webcam_target, 
                    'count': count, 
                    'timestamp': time.time()
                })
                
                st.image(result_img, caption=f"Processed Webcam Image", use_container_width=True)
                
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.markdown(f"**Target ({webcam_target}) Detected**<br><span class='value'>{count}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence**<br><span class='value'>{confidence:.2%}</span>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_stat2:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.markdown("**All Detected Objects**", unsafe_allow_html=True)
                    if detected_classes:
                        for class_name, data in detected_classes.items():
                            st.markdown(f"{class_name}: <span class='value'>{data['count']}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("No objects detected")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown(
                    get_binary_file_downloader_html(
                        image_to_bytes(result_img), 
                        "Processed Webcam Image", 
                        f"webcam_detection_{int(time.time())}.png"
                    ), 
                    unsafe_allow_html=True
                )
                
                if detected_classes and len(detected_classes) > 1:
                    st.subheader("Detection Details")
                    chart = create_class_distribution_chart(detected_classes)
                    st.plotly_chart(chart, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Click the camera button above to capture an image for detection.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Detection Analytics")
        if st.session_state.detection_history:
            timestamps = [item['timestamp'] for item in st.session_state.detection_history]
            counts = [item['count'] for item in st.session_state.detection_history]
            fig = go.Figure(data=go.Scatter(
                x=[time.strftime('%H:%M:%S', time.localtime(t)) for t in timestamps],
                y=counts, 
                mode='lines+markers', 
                marker=dict(size=10)
            ))
            fig.update_layout(
                title="Detection History", 
                xaxis_title="Time", 
                yaxis_title="Number of Objects Detected"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No detection history available yet.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Sample Detections")
        gallery = generate_gallery_images()
        for item in gallery:
            st.markdown(
                f'<div><img src="{item["image"]}" width="100%"><p>{item["title"]}</p><small>{item["description"]}</small></div>', 
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("About AI Vision Lab")
        st.markdown("""
        AI Vision Lab is built with:
        - YOLOv11 for object detection
        - Streamlit for the interface
        - Custom 3D visualization
        
        Features:
        - Image and webcam detection
        - Customizable visualization
        - Real-time analytics
        """)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()