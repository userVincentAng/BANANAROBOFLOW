import os
import io
import cv2
import math
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# -----------------------------
# Model loading (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    banana_model = YOLO("runs/detect/train/weights/best.pt")
    coco_model = YOLO("yolov8n.pt")
    return banana_model, coco_model

# -----------------------------
# Drawing helpers (adapted)
# -----------------------------
def create_clear_label(image, text, position, bg_color, text_color=(255, 255, 255),
                       font_scale=0.7, thickness=2, padding=8):
    x, y = position
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    bg_x1 = max(0, x - padding)
    bg_y1 = max(0, y - th - padding)
    bg_x2 = min(image.shape[1], x + tw + padding)
    bg_y2 = min(image.shape[0], y + baseline + padding)
    overlay = image.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
    return bg_x2, bg_y2

def draw_enhanced_bounding_box(image, x1, y1, x2, y2, color, label="", confidence=0.0, thickness=3,
                               high_contrast_labels=True):
    if isinstance(color, str) and color.startswith('#'):
        color = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    marker_size = 15
    cv2.line(image, (x1, y1), (x1 + marker_size, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + marker_size), color, thickness)
    cv2.line(image, (x2, y1), (x2 - marker_size, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + marker_size), color, thickness)
    cv2.line(image, (x1, y2), (x1 + marker_size, y2), color, thickness)
    cv2.line(image, (x1, y2), (x1, y2 - marker_size), color, thickness)
    cv2.line(image, (x2, y2), (x2 - marker_size, y2), color, thickness)
    cv2.line(image, (x2, y2), (x2, y2 - marker_size), color, thickness)

    if label:
        full_label = label
        if confidence > 0:
            full_label += f" ({confidence:.2f})"
        if high_contrast_labels:
            text_color = (255, 255, 255)
            bg_color = (0, 0, 0)
        else:
            text_color = (255, 255, 255)
            bg_color = color
        label_x = x1
        label_y = y1 - 8 if y1 - 8 > 10 else y1 + 24
        create_clear_label(image, full_label, (label_x, label_y), bg_color, text_color,
                           font_scale=0.7, thickness=2)
    return image

def get_ripeness_color(score):
    colors = {
        1: "#00FF00",
        2: "#ADFF2F",
        3: "#FFFF00",
        4: "#FFA500",
        5: "#8B4513"
    }
    return colors.get(int(round(score)), "#FFFF00")

# -----------------------------
# Ripeness analysis
# -----------------------------
def calculate_age_from_brown_percentage(brown_pct):
    if brown_pct < 0.1:
        return 0.0
    decade = min(9, int(brown_pct // 10))
    fraction = (brown_pct % 10) / 10.0
    age_days = decade + fraction
    age_days = min(10.0, age_days)
    return round(age_days, 1)

def get_ripeness_stage_from_age(age_days):
    if age_days < 1.0:
        return "Green-Yellow (0-0.9 days)"
    elif age_days < 9.0:
        return f"Yellow-Brown ({int(age_days)}-{int(age_days)}.9 days)"
    else:
        return "Overripe (9-10 days)"

def calculate_ripeness_score_improved(yellow_pct, green_pct, brown_pct, spot_pct, spot_count, avg_size):
    score = 1.0
    if green_pct > 50:
        score = 1.0
    elif green_pct > 20:
        score = 1.5
    else:
        if yellow_pct > 60:
            score = 3.0
        elif yellow_pct > 35:
            score = 2.5
        else:
            score = 2.0

    if brown_pct > 40:
        score += 2.0
    elif brown_pct > 20:
        score += 1.2
    elif brown_pct > 8:
        score += 0.6

    if spot_pct > 20 or spot_count > 25:
        score += 1.0
    elif spot_pct > 8 or spot_count > 8:
        score += 0.6

    score = max(1.0, min(5.0, score))
    return round(score, 1)

def calculate_sugar_content(detections, total_weight, formula_type="basic"):
    """
    Calculate sugar content based on banana age and weight
    Formula: sugar content = (Total Age * Weight) / 100
    """
    if not detections:
        return 0.0, "No bananas detected", 0.0, 0
    
    # Calculate total age (sum of all banana ages)
    total_age = sum([det['analysis']['age_days'] for det in detections])
    banana_count = len(detections)
    
    # Calculate total weight
    total_weight_grams = total_weight * banana_count
    
    if formula_type == "basic":
        # Basic formula: (Total Age × Total Weight) / 100
        sugar_content = (total_age * total_weight_grams) / 100
        formula_explanation = f"({total_age} days × {total_weight_grams}g) / 100 = {sugar_content:.1f}g"
    
    else:
        # Advanced formula with ripeness factor
        avg_ripeness = np.mean([det['analysis']['ripeness_score'] for det in detections])
        ripeness_factor = 0.5 + (avg_ripeness / 10)  # Factor from 0.6 to 1.0
        sugar_content = (total_age * total_weight_grams * ripeness_factor) / 100
        formula_explanation = f"({total_age} days × {total_weight_grams}g × {ripeness_factor:.2f}) / 100 = {sugar_content:.1f}g"
    
    return sugar_content, formula_explanation, total_age, banana_count

def analyze_banana_ripeness(cropped_region, sensitivity):
    h, w = cropped_region.shape[:2]
    if h == 0 or w == 0:
        return {
            'green_percent': 0, 'yellow_percent': 0, 'brown_percent': 0,
            'age_days': 0, 'ripeness_stage': "Unknown",
            'spots_mask': np.zeros((1, 1), np.uint8), 'spot_contours': [],
            'ripeness_score': 1, 'spot_percent': 0, 'spot_count': 0, 'avg_spot_size': 0
        }

    hsv = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    hsv_eq = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_eq, lower_green, upper_green)

    lower_yellow = np.array([15, 60, 80])
    upper_yellow = np.array([45, 255, 255])
    yellow_mask = cv2.inRange(hsv_eq, lower_yellow, upper_yellow)

    lower_brown = np.array([5, 50, 20])
    upper_brown = np.array([25, 255, 160])
    brown_mask = cv2.inRange(hsv_eq, lower_brown, upper_brown)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k, iterations=1)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, k, iterations=1)
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, k, iterations=1)

    combined_mask = cv2.bitwise_or(green_mask, yellow_mask)
    combined_mask = cv2.bitwise_or(combined_mask, brown_mask)
    banana_area = combined_mask
    banana_area_pixels = np.count_nonzero(banana_area)
    total_pixels = h * w

    if banana_area_pixels < 0.02 * total_pixels:
        banana_area = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
        _, banana_area = cv2.threshold(banana_area, 10, 255, cv2.THRESH_BINARY)
        banana_area_pixels = np.count_nonzero(banana_area)

    gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    blocksize = 11 if min(h, w) > 100 else 7
    c = max(2, int(6 - sensitivity * 5))
    spots = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, blocksize, c)
    spots = cv2.bitwise_and(spots, spots, mask=banana_area.astype(np.uint8))

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, kernel_small, iterations=1)
    spots = cv2.morphologyEx(spots, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    contours, _ = cv2.findContours(spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    spot_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 8:
            continue
        filtered_contours.append(cnt)
        spot_areas.append(area)

    spot_count = len(filtered_contours)
    avg_spot_size = float(np.mean(spot_areas)) if spot_areas else 0.0
    spot_pixels = sum(spot_areas)
    spot_percent = (spot_pixels / max(1, banana_area_pixels)) * 100

    green_count = np.count_nonzero(green_mask & banana_area)
    yellow_count = np.count_nonzero(yellow_mask & banana_area)
    brown_count = np.count_nonzero(brown_mask & banana_area)

    denom = max(1, banana_area_pixels)
    green_percent = green_count / denom * 100
    yellow_percent = yellow_count / denom * 100
    brown_percent = brown_count / denom * 100

    age_days = calculate_age_from_brown_percentage(brown_percent)
    ripeness_stage = get_ripeness_stage_from_age(age_days)
    ripeness_score = calculate_ripeness_score_improved(yellow_percent, green_percent,
                                                       brown_percent, spot_percent,
                                                       spot_count, avg_spot_size)

    return {
        'green_percent': green_percent,
        'yellow_percent': yellow_percent,
        'brown_percent': brown_percent,
        'age_days': age_days,
        'ripeness_stage': ripeness_stage,
        'spots_mask': spots,
        'spot_contours': filtered_contours,
        'banana_mask': banana_area,
        'ripeness_score': ripeness_score,
        'spot_percent': spot_percent,
        'spot_count': spot_count,
        'avg_spot_size': avg_spot_size
    }

# -----------------------------
# Detection flows
# -----------------------------
def process_banana_detection(img_bgr, model, sensitivity, show_spots, high_contrast_labels, weight_per_banana):
    results = model.predict(img_bgr, verbose=False)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    banana_detections = []
    confidence_threshold = max(0.3, sensitivity)

    for r in results:
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            if confidence < confidence_threshold:
                continue
            cropped = img_bgr[y1:y2, x1:x2].copy()
            if cropped.size == 0:
                continue
            analysis = analyze_banana_ripeness(cropped, sensitivity)

            # Calculate individual sugar content
            individual_sugar = (analysis['age_days'] * weight_per_banana) / 100

            ripeness_stage = analysis['ripeness_stage']
            color = get_ripeness_color(analysis['ripeness_score'])

            # Enhanced label with sugar information
            label_text = f"Banana {i + 1}: {analysis['age_days']}d | Sugar: {individual_sugar:.1f}g"
            draw_enhanced_bounding_box(img_rgb, x1, y1, x2, y2, color, label_text, confidence,
                                       high_contrast_labels=high_contrast_labels)

            if show_spots:
                for cnt in analysis['spot_contours']:
                    cnt_shifted = cnt + np.array([[x1, y1]])
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00']) + x1
                        cy = int(M['m01'] / M['m00']) + y1
                        cv2.circle(img_rgb, (cx, cy), 4, (255, 0, 0), -1)

            banana_detections.append({
                'bbox': (x1, y1, x2, y2),
                'analysis': analysis,
                'ripeness_stage': ripeness_stage,
                'confidence': confidence,
                'sugar_content': individual_sugar
            })

    # Return annotated image directly in memory
    is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    annotated_img = Image.open(io.BytesIO(buffer))

    return {
        'image': annotated_img,
        'detections': banana_detections,
        'mode': 'banana'
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Banana Ripeness & Sugar Content Detection", layout="wide")
st.title("🍌 Banana Ripeness & Sugar Content Detection")

# Initialize session state for weight persistence
if 'weight_per_banana' not in st.session_state:
    st.session_state.weight_per_banana = 120.0
if 'sugar_formula' not in st.session_state:
    st.session_state.sugar_formula = "Basic: (Total Age × Weight) / 100"

with st.sidebar:
    st.header("🎛️ Controls")
    sensitivity = st.slider("Detection Sensitivity", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
    show_spots = st.checkbox("Show Spots", value=True)
    high_contrast_labels = st.checkbox("High Contrast Labels", value=True)
    
    st.subheader("🍬 Nutritional Analysis")
    weight_per_banana = st.number_input(
        "Average Weight per Banana (grams)", 
        min_value=1.0, 
        max_value=500.0, 
        value=st.session_state.weight_per_banana,
        step=1.0,
        help="Typical banana weight: 100-150g"
    )
    
    # Update session state when value changes
    if weight_per_banana != st.session_state.weight_per_banana:
        st.session_state.weight_per_banana = weight_per_banana
    
    with st.expander("Advanced Sugar Parameters"):
        sugar_formula = st.selectbox(
            "Sugar Content Formula",
            options=["Basic: (Total Age × Weight) / 100", "Advanced: (Total Age × Weight × Ripeness Factor) / 100"],
            index=0 if st.session_state.sugar_formula.startswith("Basic") else 1,
            help="Choose the calculation method"
        )
        if sugar_formula != st.session_state.sugar_formula:
            st.session_state.sugar_formula = sugar_formula

# Load models
banana_model, coco_model = load_models()

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image of bananas", type=["jpg", "jpeg", "png", "bmp"])

# Layout columns
col_left, col_right = st.columns([3, 2])

with col_left:
    image_placeholder = st.empty()

with col_right:
    st.subheader("📊 Detection Results")
    result_box = st.empty()

# Detection button
detect_clicked = st.button("🔍 Detect Ripeness & Sugar Content", type="primary")

# Image processing
image_bgr = None
if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_bgr is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_placeholder.image(image_rgb, caption="Uploaded Image", use_container_width=True)
    except Exception as e:
        st.error(f"❌ Failed to load image: {e}")

# Detection processing
if detect_clicked:
    if image_bgr is None:
        st.warning("⚠️ Please upload an image first.")
    else:
        with st.spinner("🔬 Analyzing bananas..."):
            # Determine formula type
            formula_type = "basic" if "Basic" in st.session_state.sugar_formula else "advanced"
            
            # Process detection
            results = process_banana_detection(
                image_bgr, banana_model, sensitivity, show_spots, high_contrast_labels, weight_per_banana
            )
            
            # Display annotated result
            image_placeholder.image(results['image'], caption="Detection Result", use_container_width=True)

            if results.get('detections'):
                # Calculate total sugar content
                sugar_content, formula_explanation, total_age, banana_count = calculate_sugar_content(
                    results['detections'], weight_per_banana, formula_type
                )
                
                # Enhanced results display with sugar content
                lines = [
                    "🍌 BANANA RIPENESS & SUGAR ANALYSIS", 
                    "=" * 50, 
                    ""
                ]
                
                # Nutritional summary
                lines.extend([
                    "📊 NUTRITIONAL SUMMARY",
                    f"• Bananas Detected: {banana_count}",
                    f"• Total Age: {total_age:.1f} days",
                    f"• Average Weight per Banana: {weight_per_banana:.1f}g",
                    f"• Total Weight: {weight_per_banana * banana_count:.1f}g",
                    f"• Estimated Total Sugar Content: {sugar_content:.1f}g",
                    f"• Calculation: {formula_explanation}",
                    ""
                ])
                
                # Sugar context information
                lines.extend([
                    "💡 SUGAR CONTEXT",
                    f"• Equivalent to {sugar_content / 4:.1f} teaspoons of sugar",
                    f"• {sugar_content / 25 * 100:.1f}% of recommended daily intake (25g)",
                    ""
                ])
                
                # Individual banana analysis
                lines.append("🔍 INDIVIDUAL BANANA ANALYSIS")
                for i, det in enumerate(results['detections']):
                    a = det['analysis']
                    lines.extend([
                        f"Banana {i + 1} Results:",
                        f"  • Age: {a['age_days']} days",
                        f"  • Sugar Content: {det['sugar_content']:.1f}g",
                        f"  • Ripeness Score: {a['ripeness_score']}/5.0",
                        f"  • Stage: {det['ripeness_stage']}",
                        f"  • Brown Percentage: {a['brown_percent']:.1f}%",
                        f"  • Green: {a['green_percent']:.1f}%",
                        f"  • Yellow: {a['yellow_percent']:.1f}%",
                        f"  • Spot Coverage: {a['spot_percent']:.1f}%",
                        f"  • Spot Count: {a['spot_count']}",
                        ""
                    ])
                
                # Display results
                result_box.code("\n".join(lines))
                
                # Additional visualizations
                with st.expander("🍬 Sugar Content Visualization", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Sugar content gauge
                        max_sugar = 50
                        sugar_percentage = min(100, (sugar_content / max_sugar) * 100)
                        st.metric("Total Sugar", f"{sugar_content:.1f}g")
                        st.progress(sugar_percentage/100)
                    
                    with col2:
                        # Daily intake percentage
                        daily_percentage = (sugar_content / 25) * 100
                        st.metric("Daily Intake", f"{daily_percentage:.1f}%")
                        st.progress(min(100, daily_percentage)/100)
                    
                    with col3:
                        # Average per banana
                        avg_sugar = sugar_content / banana_count if banana_count > 0 else 0
                        st.metric("Avg per Banana", f"{avg_sugar:.1f}g")
                
            else:
                result_box.info("ℹ️ No bananas detected. Try adjusting sensitivity or use a clearer image.")

# Add footer with information
st.markdown("---")
st.markdown(
    """
    **💡 Tips for best results:**
    - Use clear, well-lit images of bananas
    - Ensure bananas are visible and not overlapping too much
    - Adjust sensitivity if detection is too strict/lenient
    - Typical banana weight is 100-150g for accurate sugar estimation
    """
)