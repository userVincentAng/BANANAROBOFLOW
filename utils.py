import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
import sys
import torch

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
        self.peel_multiplier = 0.6338  # Multiplier when peel is on
        
    def predict_sugar_percentage(self, age_days):
        """Predict sugar percentage (g/100ml) based on age using the regression model"""
        validated_age = self.validate_banana_age(age_days)
        return self.intercept + self.slope * validated_age
    
    def calculate_sugar_content(self, age_days, weight_grams, has_peel=True):
        """Calculate total sugar content in grams based on age and weight"""
        sugar_percentage = self.predict_sugar_percentage(age_days)
        # Convert percentage to grams: (sugar_percentage/100) * weight_grams
        sugar_content = (sugar_percentage / 100) * weight_grams
        
        # Apply peel multiplier if peel is on
        if has_peel:
            sugar_content *= self.peel_multiplier
            
        return sugar_content
    
    def predict_sugar_projection(self, current_age_days, weight_grams, has_peel=True, days_ahead=10):
        """Predict sugar content for the next N days starting from current age"""
        projection = []
        for days in range(days_ahead + 1):
            future_age = current_age_days + days
            validated_age = self.validate_banana_age(future_age)
            sugar_percentage = self.predict_sugar_percentage(validated_age)
            sugar_content = self.calculate_sugar_content(validated_age, weight_grams, has_peel)
            
            projection.append({
                'day': days,
                'age_days': future_age,
                'validated_age': validated_age,
                'sugar_percentage': sugar_percentage,
                'sugar_content': sugar_content,
                'is_current': days == 0
            })
        
        return projection
    
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
            # Collect safe globals, handling optional imports gracefully
            safe_objs = [torch.nn.modules.container.Sequential, DetectionModel]
            try:
                from ultralytics.nn.modules.conv import Conv  # type: ignore
                safe_objs.append(Conv)
            except Exception:
                pass
            try:
                from ultralytics.nn.modules.conv import DWConv  # type: ignore
                safe_objs.append(DWConv)
            except Exception:
                pass
            try:
                from ultralytics.nn.modules.conv import C2f  # type: ignore
                safe_objs.append(C2f)
            except Exception:
                pass
            try:
                from ultralytics.nn.modules.conv import Bottleneck  # type: ignore
                safe_objs.append(Bottleneck)
            except Exception:
                pass
            try:
                from ultralytics.nn.modules.conv import SPPF  # type: ignore
                safe_objs.append(SPPF)
            except Exception:
                pass

            add_safe_globals(safe_objs)
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

def calculate_ripeness_score_impro