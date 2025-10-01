import cv2
import numpy as np
from PIL import Image
import io

from utils import analyze_banana_ripeness, BananaSugarModel

def create_clear_label(image, text, position, bg_color, text_color=(255, 255, 255),
                       font_scale=0.7, thickness=2, padding=8):
    """Create a clear text label with background"""
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
    """Draw enhanced bounding box with markers and label"""
    if isinstance(color, str) and color.startswith('#'):
        color = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    
    # Draw main rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw corner markers
    marker_size = 15
    cv2.line(image, (x1, y1), (x1 + marker_size, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + marker_size), color, thickness)
    cv2.line(image, (x2, y1), (x2 - marker_size, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + marker_size), color, thickness)
    cv2.line(image, (x1, y2), (x1 + marker_size, y2), color, thickness)
    cv2.line(image, (x1, y2), (x1, y2 - marker_size), color, thickness)
    cv2.line(image, (x2, y2), (x2 - marker_size, y2), color, thickness)
    cv2.line(image, (x2, y2), (x2, y2 - marker_size), color, thickness)

    # Draw label
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
    """Get color code for ripeness score"""
    colors = {
        1: "#00FF00",  # Green
        2: "#ADFF2F",  # Green-Yellow
        3: "#FFFF00",  # Yellow
        4: "#FFA500",  # Orange
        5: "#8B4513"   # Brown
    }
    return colors.get(int(round(score)), "#FFFF00")

def process_banana_detection(img_bgr, model, sensitivity, show_spots, high_contrast_labels, weight_per_banana, has_peel=True):
    """Main detection processing function with visualization - UPDATED to accept has_peel parameter"""
    results = model.predict(img_bgr, verbose=False)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    banana_detections = []
    confidence_threshold = max(0.3, sensitivity)
    sugar_model = BananaSugarModel()

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

            # Calculate sugar content using scientific model with peel option
            sugar_percentage = sugar_model.predict_sugar_percentage(analysis['age_days'])
            individual_sugar = sugar_model.calculate_sugar_content(analysis['age_days'], weight_per_banana, has_peel)

            ripeness_stage = analysis['ripeness_stage']
            color = get_ripeness_color(analysis['ripeness_score'])

            # Enhanced label with scientific sugar information and peel status
            peel_status = "peeled" if not has_peel else "with peel"
            label_text = f"Banana {i + 1}: {analysis['age_days']}d | Sugar: {sugar_percentage:.1f}% ({peel_status})"
            draw_enhanced_bounding_box(img_rgb, x1, y1, x2, y2, color, label_text, confidence,
                                       high_contrast_labels=high_contrast_labels)

            # Draw spots if enabled
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
                'sugar_percentage': sugar_percentage,
                'sugar_content': individual_sugar,
                'has_peel': has_peel
            })

    # Convert back to PIL Image for Streamlit
    is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    annotated_img = Image.open(io.BytesIO(buffer))

    return {
        'image': annotated_img,
        'detections': banana_detections,
        'mode': 'banana'
    }