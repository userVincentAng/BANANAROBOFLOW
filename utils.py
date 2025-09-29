import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
import sys

import streamlit as st

# -----------------------------
# Scientific Sugar Model
# -----------------------------
class BananaSugarModel:
    def __init__(self):
        # Based on dataset regression: Sugar = 25.99 + 0.707 * Age
        self.intercept = 25.99285714285715
        self.slope = 0.707142857142857
        self.r_squared = 0.8672  # High correlation!
        self.standard_error = 0.6548
        
    def predict_sugar_percentage(self, age_days):
        """Predict sugar percentage (g/100ml) based on age using the regression model"""
        validated_age = self.validate_banana_age(age_days)
        return self.intercept + self.slope * validated_age
    
    def calculate_sugar_content(self, age_days, weight_grams):
        """Calculate total sugar content in grams based on age and weight"""
        sugar_percentage = self.predict_sugar_percentage(age_days)
        # Convert percentage to grams: (sugar_percentage/100) * weight_grams
        return (sugar_percentage / 100) * weight_grams
    
    def validate_banana_age(self, age_days):
        """Validate age against dataset range with interpolation for extremes"""
        if age_days < 1:
            return 1.0  # Minimum from dataset
        elif age_days > 8:
            # Extrapolate with caution for older bananas
            return min(10.0, age_days)
        return age_days

# -----------------------------
# Model loading (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    """Load YOLO models with caching"""
    try:
        # PyTorch 2.6+ defaults torch.load(weights_only=True). Allowlist Ultralytics class for safe loading.
        try:
            # Defer imports so this remains compatible on older torch versions
            from torch.serialization import add_safe_globals  # type: ignore
            from ultralytics.nn.tasks import DetectionModel  # type: ignore
            add_safe_globals([DetectionModel])
        except Exception:
            # Best-effort: if add_safe_globals is unavailable, proceed as usual
            pass

        banana_model = YOLO("runs/detect/train/weights/best.pt")
        coco_model = YOLO("yolov8n.pt")
        return banana_model, coco_model
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        return None, None

# -----------------------------
# Ripeness Analysis Functions
# -----------------------------
def calculate_age_from_brown_percentage(brown_pct):
    """Calculate banana age from brown percentage"""
    if brown_pct < 0.1:
        return 0.0
    decade = min(9, int(brown_pct // 10))
    fraction = (brown_pct % 10) / 10.0
    age_days = decade + fraction
    age_days = min(10.0, age_days)
    return round(age_days, 1)

def get_ripeness_stage_from_age(age_days):
    """Get ripeness stage description from age"""
    if age_days < 1.0:
        return "Green-Yellow (0-0.9 days)"
    elif age_days < 9.0:
        return f"Yellow-Brown ({int(age_days)}-{int(age_days)}.9 days)"
    else:
        return "Overripe (9-10 days)"

def calculate_ripeness_score_improved(yellow_pct, green_pct, brown_pct, spot_pct, spot_count, avg_size):
    """Calculate comprehensive ripeness score"""
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

def calculate_sugar_content(detections, total_weight, formula_type="scientific"):
    """
    Calculate sugar content using scientific regression model or fallback methods
    """
    if not detections:
        if formula_type == "scientific":
            return 0.0, "No bananas detected", 0.0, 0, []
        else:
            return 0.0, "No bananas detected", 0.0, 0
    
    banana_count = len(detections)
    
    if formula_type == "scientific":
        sugar_model = BananaSugarModel()
        total_sugar = 0.0
        sugar_breakdown = []
        
        for i, det in enumerate(detections):
            age_days = det['analysis']['age_days']
            sugar_percentage = sugar_model.predict_sugar_percentage(age_days)
            individual_sugar = sugar_model.calculate_sugar_content(age_days, total_weight)
            total_sugar += individual_sugar
            
            sugar_breakdown.append({
                'banana': i + 1,
                'age_days': age_days,
                'sugar_percentage': sugar_percentage,
                'sugar_grams': individual_sugar
            })
        
        formula_explanation = f"Sugar% = 25.99 + 0.707 × Age (R²=0.867)"
        total_age = sum([det['analysis']['age_days'] for det in detections])
        
        return total_sugar, formula_explanation, total_age, banana_count, sugar_breakdown
        
    else:
        total_age = sum([det['analysis']['age_days'] for det in detections])
        total_weight_grams = total_weight * banana_count
        
        if formula_type == "basic":
            sugar_content = (total_age * total_weight_grams) / 100
            formula_explanation = f"({total_age} days × {total_weight_grams}g) / 100 = {sugar_content:.1f}g"
        else:
            avg_ripeness = np.mean([det['analysis']['ripeness_score'] for det in detections])
            ripeness_factor = 0.5 + (avg_ripeness / 10)
            sugar_content = (total_age * total_weight_grams * ripeness_factor) / 100
            formula_explanation = f"({total_age} days × {total_weight_grams}g × {ripeness_factor:.2f}) / 100 = {sugar_content:.1f}g"
        
        return sugar_content, formula_explanation, total_age, banana_count

def analyze_banana_ripeness(cropped_region, sensitivity):
    """Main ripeness analysis function"""
    h, w = cropped_region.shape[:2]
    if h == 0 or w == 0:
        return create_empty_analysis_result()

    # Image preprocessing
    hsv = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2LAB)

    # Enhance contrast
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    hsv_eq = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV)

    # Color segmentation
    green_mask = create_color_mask(hsv_eq, [35, 40, 40], [85, 255, 255])
    yellow_mask = create_color_mask(hsv_eq, [15, 60, 80], [45, 255, 255])
    brown_mask = create_color_mask(hsv_eq, [5, 50, 20], [25, 255, 160])

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Combine masks and find banana area
    combined_mask = cv2.bitwise_or(green_mask, yellow_mask)
    combined_mask = cv2.bitwise_or(combined_mask, brown_mask)
    banana_area = combined_mask
    banana_area_pixels = np.count_nonzero(banana_area)
    total_pixels = h * w

    # Fallback if banana area is too small
    if banana_area_pixels < 0.02 * total_pixels:
        banana_area = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
        _, banana_area = cv2.threshold(banana_area, 10, 255, cv2.THRESH_BINARY)
        banana_area_pixels = np.count_nonzero(banana_area)

    # Spot detection
    gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    blocksize = 11 if min(h, w) > 100 else 7
    c = max(2, int(6 - sensitivity * 5))
    spots = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, blocksize, c)
    spots = cv2.bitwise_and(spots, spots, mask=banana_area.astype(np.uint8))

    # Clean up spots
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, kernel_small, iterations=1)
    spots = cv2.morphologyEx(spots, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    # Analyze contours and calculate metrics
    contours, _ = cv2.findContours(spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours, spot_areas = filter_contours(contours)
    
    spot_count = len(filtered_contours)
    avg_spot_size = float(np.mean(spot_areas)) if spot_areas else 0.0
    spot_pixels = sum(spot_areas)
    spot_percent = (spot_pixels / max(1, banana_area_pixels)) * 100

    # Calculate color percentages
    green_count = np.count_nonzero(green_mask & banana_area)
    yellow_count = np.count_nonzero(yellow_mask & banana_area)
    brown_count = np.count_nonzero(brown_mask & banana_area)

    denom = max(1, banana_area_pixels)
    green_percent = green_count / denom * 100
    yellow_percent = yellow_count / denom * 100
    brown_percent = brown_count / denom * 100

    # Final analysis
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
# Helper Functions
# -----------------------------
def create_color_mask(hsv_image, lower_bound, upper_bound):
    """Create color mask for given HSV range"""
    return cv2.inRange(hsv_image, np.array(lower_bound), np.array(upper_bound))

def filter_contours(contours, min_area=8):
    """Filter contours by area"""
    filtered = []
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            filtered.append(cnt)
            areas.append(area)
    return filtered, areas

def create_empty_analysis_result():
    """Create empty analysis result for error cases"""
    return {
        'green_percent': 0, 'yellow_percent': 0, 'brown_percent': 0,
        'age_days': 0, 'ripeness_stage': "Unknown",
        'spots_mask': np.zeros((1, 1), np.uint8), 'spot_contours': [],
        'ripeness_score': 1, 'spot_percent': 0, 'spot_count': 0, 'avg_spot_size': 0
    }