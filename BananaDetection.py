import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os
import math

class BananaAgeDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Banana Ripeness Detection System")
        self.root.geometry("1200x800")
        
        # Load models
        self.banana_model = YOLO("runs/detect/train/weights/best.pt")
        self.coco_model = YOLO("yolov8n.pt")
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        # (kept your GUI code mostly unchanged)
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        title_label = tk.Label(main_frame, text="Banana Ripeness Detection System", 
                              font=("Arial", 18, "bold"))
        title_label.pack(pady=10)
        
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(control_frame, text="Detection Mode:", font=("Arial", 11)).grid(row=0, column=0, padx=5)
        self.model_var = tk.StringVar(value="Banana Ripeness Detection")
        model_options = ["Banana Ripeness Detection", "General Object Detection"]
        model_menu = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                 values=model_options, state="readonly", width=25)
        model_menu.grid(row=0, column=1, padx=5)
        
        param_frame = tk.Frame(control_frame)
        param_frame.grid(row=0, column=2, padx=30)
        
        tk.Label(param_frame, text="Detection Sensitivity:", font=("Arial", 10)).pack()
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        sensitivity_scale = tk.Scale(param_frame, variable=self.sensitivity_var, 
                                    from_=0.1, to=0.9, resolution=0.1, orient=tk.HORIZONTAL, 
                                    length=150, showvalue=True)
        sensitivity_scale.pack()
        
        display_frame = tk.Frame(control_frame)
        display_frame.grid(row=0, column=3, padx=20)
        
        self.show_spots_var = tk.BooleanVar(value=True)
        self.show_spots_cb = tk.Checkbutton(display_frame, text="Show Spots", 
                                           variable=self.show_spots_var, font=("Arial", 10))
        self.show_spots_cb.pack()
        
        self.high_contrast_var = tk.BooleanVar(value=True)
        self.high_contrast_cb = tk.Checkbutton(display_frame, text="High Contrast Labels", 
                                              variable=self.high_contrast_var, font=("Arial", 10))
        self.high_contrast_cb.pack()
        
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=15)
        
        self.select_btn = tk.Button(button_frame, text="Select Image", 
                                   command=self.select_image, 
                                   font=("Arial", 12), bg="#4CAF50", fg="white", 
                                   width=15, height=1)
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        self.camera_btn = tk.Button(button_frame, text="Use Camera", 
                                   command=self.use_camera, 
                                   font=("Arial", 12), bg="#2196F3", fg="white", 
                                   width=15, height=1)
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        results_frame = tk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        image_container = tk.Frame(results_frame)
        image_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(image_container, bg="white")
        self.scrollbar_y = tk.Scrollbar(image_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar_x = tk.Scrollbar(image_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_label = tk.Label(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_label, anchor="nw")
        
        info_frame = tk.Frame(results_frame, relief=tk.GROOVE, bd=2, width=350)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        info_frame.pack_propagate(False)
        
        tk.Label(info_frame, text="Detection Results", font=("Arial", 14, "bold")).pack(pady=15)
        
        self.results_text = tk.Text(info_frame, height=25, width=40, font=("Arial", 10),
                                   wrap=tk.WORD, padx=10, pady=10)
        scrollbar_text = tk.Scrollbar(info_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar_text.set)
        scrollbar_text.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar(value="Ready to detect bananas...")
        status_bar = tk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, 
                             anchor=tk.W, font=("Arial", 9))
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    # ---------- drawing helpers (unchanged) ----------
    def create_clear_label(self, image, text, position, bg_color, text_color=(255,255,255),
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

    def draw_enhanced_bounding_box(self, image, x1, y1, x2, y2, color, label="", confidence=0.0, thickness=3):
        if isinstance(color, str) and color.startswith('#'):
            color = tuple(int(color[i:i+2], 16) for i in (1,3,5))
        cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness)
        marker_size = 15
        cv2.line(image, (x1, y1), (x1+marker_size, y1), color, thickness)
        cv2.line(image, (x1, y1), (x1, y1+marker_size), color, thickness)
        cv2.line(image, (x2, y1), (x2-marker_size, y1), color, thickness)
        cv2.line(image, (x2, y1), (x2, y1+marker_size), color, thickness)
        cv2.line(image, (x1, y2), (x1+marker_size, y2), color, thickness)
        cv2.line(image, (x1, y2), (x1, y2-marker_size), color, thickness)
        cv2.line(image, (x2, y2), (x2-marker_size, y2), color, thickness)
        cv2.line(image, (x2, y2), (x2, y2-marker_size), color, thickness)

        if label:
            full_label = label
            if confidence > 0:
                full_label += f" ({confidence:.2f})"
            if self.high_contrast_var.get():
                text_color = (255,255,255)
                bg_color = (0,0,0)
            else:
                text_color = (255,255,255)
                bg_color = color
            label_x = x1
            label_y = y1 - 8 if y1 - 8 > 10 else y1 + 24
            self.create_clear_label(image, full_label, (label_x, label_y), bg_color, text_color, font_scale=0.7, thickness=2)
        return image

    # ---------- UPDATED analysis with new age scale ----------
    def analyze_banana_ripeness(self, cropped_region):
        """
        Analyze banana ripeness and return age in days based on brown percentage
        Scale: 0-10 days where each 10% brown = 1 day
        """
        h, w = cropped_region.shape[:2]
        
        # Quick safety check
        if h == 0 or w == 0:
            return {
                'green_percent': 0, 'yellow_percent': 0, 'brown_percent': 0,
                'age_days': 0, 'ripeness_stage': "Unknown",
                'spots_mask': np.zeros((1,1), np.uint8), 'spot_contours': [],
                'ripeness_score': 1, 'spot_percent': 0, 'spot_count': 0, 'avg_spot_size': 0
            }

        # Convert to HSV and LAB
        hsv = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2LAB)

        # Apply CLAHE on L channel to improve contrast
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        hsv_eq = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV)

        # Color ranges for banana ripening stages
        # Green (unripe)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_eq, lower_green, upper_green)

        # Yellow (ripening)
        lower_yellow = np.array([15, 60, 80])
        upper_yellow = np.array([45, 255, 255])
        yellow_mask = cv2.inRange(hsv_eq, lower_yellow, upper_yellow)

        # Brown (overripe)
        lower_brown = np.array([5, 50, 20])
        upper_brown = np.array([25, 255, 160])
        brown_mask = cv2.inRange(hsv_eq, lower_brown, upper_brown)

        # Clean masks
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k, iterations=1)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, k, iterations=1)
        brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, k, iterations=1)

        # Use combined mask as banana area
        combined_mask = cv2.bitwise_or(green_mask, yellow_mask)
        combined_mask = cv2.bitwise_or(combined_mask, brown_mask)
        banana_area = combined_mask
        banana_area_pixels = np.count_nonzero(banana_area)
        total_pixels = h * w
        
        if banana_area_pixels < 0.02 * total_pixels:
            banana_area = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
            _, banana_area = cv2.threshold(banana_area, 10, 255, cv2.THRESH_BINARY)
            banana_area_pixels = np.count_nonzero(banana_area)

        # Spot detection (keeping your existing spot detection logic)
        gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
        sensitivity = self.sensitivity_var.get()
        blocksize = 11 if min(h,w) > 100 else 7
        c = max(2, int(6 - sensitivity*5))
        spots = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, blocksize, c)
        spots = cv2.bitwise_and(spots, spots, mask=banana_area.astype(np.uint8))

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, kernel_small, iterations=1)
        spots = cv2.morphologyEx(spots, cv2.MORPH_CLOSE, kernel_small, iterations=1)

        contours, _ = cv2.findContours(spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        spot_areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8:
                continue
            x,y,wc,hc = cv2.boundingRect(cnt)
            if area > 500:
                filtered_contours.append(cnt)
                spot_areas.append(area)
            else:
                filtered_contours.append(cnt)
                spot_areas.append(area)

        spot_count = len(filtered_contours)
        avg_spot_size = float(np.mean(spot_areas)) if spot_areas else 0.0
        spot_pixels = sum(spot_areas)
        spot_percent = (spot_pixels / max(1, banana_area_pixels)) * 100

        # Color percentages relative to banana area
        green_count = np.count_nonzero(green_mask & banana_area)
        yellow_count = np.count_nonzero(yellow_mask & banana_area)
        brown_count = np.count_nonzero(brown_mask & banana_area)
        
        denom = max(1, banana_area_pixels)
        green_percent = green_count / denom * 100
        yellow_percent = yellow_count / denom * 100
        brown_percent = brown_count / denom * 100

        # Calculate age based on brown percentage (NEW SCALE)
        age_days = self.calculate_age_from_brown_percentage(brown_percent)
        
        # Determine ripeness stage from age
        ripeness_stage = self.get_ripeness_stage_from_age(age_days)
        
        # Also calculate the original ripeness score for compatibility
        ripeness_score = self.calculate_ripeness_score_improved(yellow_percent, green_percent, 
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

    def calculate_age_from_brown_percentage(self, brown_pct):
        """
        Calculate age in days based on brown percentage using the new scale:
        - 0-9.9% brown = 0-0.9 days
        - 10-19.9% brown = 1-1.9 days
        - 20-29.9% brown = 2-2.9 days
        - 30-39.9% brown = 3-3.9 days
        - 40-49.9% brown = 4-4.9 days
        - 50-59.9% brown = 5-5.9 days
        - 60-69.9% brown = 6-6.9 days
        - 70-79.9% brown = 7-7.9 days
        - 80-89.9% brown = 8-8.9 days
        - 90-100% brown = 9-10 days
        """
        if brown_pct < 0.1:  # Handle very low percentages
            return 0.0
        
        # Calculate which decade (10% range) the brown percentage falls into
        decade = min(9, int(brown_pct // 10))  # 0-9 for 0-90% ranges
        
        # Calculate the fractional part within the decade
        fraction = (brown_pct % 10) / 10.0
        
        # Age = decade + fraction (capped at 10 days)
        age_days = decade + fraction
        age_days = min(10.0, age_days)  # Cap at 10 days
        
        return round(age_days, 1)

    def get_ripeness_stage_from_age(self, age_days):
        """Map age in days to ripeness stage based on the new scale"""
        if age_days < 1.0:
            return "Green-Yellow (0-0.9 days)"
        elif age_days < 2.0:
            return "Yellow-Brown (1-1.9 days)"
        elif age_days < 3.0:
            return "Yellow-Brown (2-2.9 days)"
        elif age_days < 4.0:
            return "Yellow-Brown (3-3.9 days)"
        elif age_days < 5.0:
            return "Yellow-Brown (4-4.9 days)"
        elif age_days < 6.0:
            return "Yellow-Brown (5-5.9 days)"
        elif age_days < 7.0:
            return "Yellow-Brown (6-6.9 days)"
        elif age_days < 8.0:
            return "Yellow-Brown (7-7.9 days)"
        elif age_days < 9.0:
            return "Yellow-Brown (8-8.9 days)"
        else:
            return "Overripe (9-10 days)"

    def calculate_ripeness_score_improved(self, yellow_pct, green_pct, brown_pct, spot_pct, spot_count, avg_size):
        """Keep your existing scoring for compatibility"""
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

    def get_ripeness_stage(self, score):
        """Keep for compatibility with existing code"""
        if score <= 1.5:
            return "Unripe (Green)"
        elif score <= 2.5:
            return "Slightly Ripe"
        elif score <= 3.5:
            return "Ripe (Perfect)"
        elif score <= 4.5:
            return "Very Ripe"
        else:
            return "Overripe"

    def get_ripeness_color(self, score):
        colors = {
            1: "#00FF00",
            2: "#ADFF2F",
            3: "#FFFF00",
            4: "#FFA500",
            5: "#8B4513"
        }
        return colors.get(int(round(score)), "#FFFF00")

    # ---------- I/O & processing ----------
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if file_path:
            self.process_image(file_path)

    def use_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not access camera")
            return
        ret, frame = cap.read()
        cap.release()
        if ret:
            temp_path = "temp_capture.jpg"
            cv2.imwrite(temp_path, frame)
            self.process_image(temp_path)

    def process_image(self, file_path):
        self.progress.start()
        self.select_btn.config(state="disabled")
        self.camera_btn.config(state="disabled")
        self.status_var.set("Processing image...")
        try:
            if self.model_var.get() == "Banana Ripeness Detection":
                results = self.process_banana_detection(file_path)
            else:
                results = self.process_general_detection(file_path)
            self.display_results(results)
            self.status_var.set("Detection completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set("Error occurred during processing")
        finally:
            self.progress.stop()
            self.select_btn.config(state="normal")
            self.camera_btn.config(state="normal")

    def process_banana_detection(self, file_path):
        results = self.banana_model(file_path)
        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        banana_detections = []
        confidence_threshold = max(0.3, self.sensitivity_var.get())

        for r in results:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                if confidence < confidence_threshold:
                    continue
                cropped = img[y1:y2, x1:x2].copy()
                if cropped.size == 0:
                    continue
                analysis = self.analyze_banana_ripeness(cropped)
                
                # Use age-based stage for display
                ripeness_stage = analysis['ripeness_stage']
                color = self.get_ripeness_color(analysis['ripeness_score'])
                
                # Update label to show age in days with brown percentage
                label_text = f"Banana {i+1}: {analysis['age_days']} days ({analysis['brown_percent']:.1f}% brown)"
                self.draw_enhanced_bounding_box(img_rgb, x1, y1, x2, y2, color, label_text, confidence)
                
                # Draw banana mask outline (optional)
                try:
                    mask = analysis.get('banana_mask')
                    if mask is not None and mask.shape[:2] == cropped.shape[:2]:
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            cnt_shifted = cnt + np.array([[x1, y1]])
                            cv2.drawContours(img_rgb, [cnt_shifted], -1, (200,200,200), 1)
                except Exception:
                    pass

                # Draw detected spots if enabled
                if self.show_spots_var.get():
                    for cnt in analysis['spot_contours']:
                        cnt_shifted = cnt + np.array([[x1, y1]])
                        M = cv2.moments(cnt)
                        if M['m00'] != 0:
                            cx = int(M['m10']/M['m00']) + x1
                            cy = int(M['m01']/M['m00']) + y1
                            cv2.circle(img_rgb, (cx, cy), 4, (255, 0, 0), -1)

                banana_detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'analysis': analysis,
                    'ripeness_stage': ripeness_stage,
                    'confidence': confidence
                })

        save_dir = "detections"
        os.makedirs(save_dir, exist_ok=True)
        result_path = os.path.join(save_dir, "banana_ripeness_result.jpg")
        cv2.imwrite(result_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        return {
            'image_path': result_path,
            'detections': banana_detections,
            'mode': 'banana'
        }

    def process_general_detection(self, file_path):
        results = self.coco_model(file_path)
        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for r in results:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                if confidence > 0.5:
                    label = f"{self.coco_model.names[class_id]} {i+1}"
                    color = (0, 255, 0)
                    self.draw_enhanced_bounding_box(img_rgb, x1, y1, x2, y2, color, label, confidence)
        save_dir = "detections"
        os.makedirs(save_dir, exist_ok=True)
        result_path = os.path.join(save_dir, "general_detection_result.jpg")
        cv2.imwrite(result_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return {
            'image_path': result_path,
            'mode': 'general'
        }

    def display_results(self, results):
        pil_img = Image.open(results['image_path'])
        max_width, max_height = 800, 600
        pil_img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.results_text.delete(1.0, tk.END)

        if results['mode'] == 'banana' and results.get('detections'):
            self.results_text.insert(tk.END, "🍌 BANANA RIPENESS ANALYSIS 🍌\n")
            self.results_text.insert(tk.END, "=" * 40 + "\n\n")
            for i, detection in enumerate(results['detections']):
                analysis = detection['analysis']
                self.results_text.insert(tk.END, f"Banana {i+1} Results:\n", "header")
                self.results_text.insert(tk.END, f"Age: {analysis['age_days']} days\n", "result")
                self.results_text.insert(tk.END, f"Brown Percentage: {analysis['brown_percent']:.1f}%\n")
                self.results_text.insert(tk.END, f"Stage: {detection['ripeness_stage']}\n")
                self.results_text.insert(tk.END, f"Green: {analysis['green_percent']:.1f}%\n")
                self.results_text.insert(tk.END, f"Yellow: {analysis['yellow_percent']:.1f}%\n")
                self.results_text.insert(tk.END, f"Spot Count: {analysis['spot_count']}\n")
                self.results_text.insert(tk.END, f"Spot Coverage: {analysis['spot_percent']:.1f}%\n\n")
                self.results_text.tag_configure("header", font=("Arial", 11, "bold"))
                self.results_text.tag_configure("result", foreground="blue")
        elif results['mode'] == 'general':
            self.results_text.insert(tk.END, "📷 GENERAL OBJECT DETECTION\n")
            self.results_text.insert(tk.END, "=" * 30 + "\n\n")
            self.results_text.insert(tk.END, "Objects detected successfully!\n")
        else:
            self.results_text.insert(tk.END, "❌ No bananas detected!\n\n")
            self.results_text.insert(tk.END, "Suggestions:\n")
            self.results_text.insert(tk.END, "• Try adjusting sensitivity slider\n")
            self.results_text.insert(tk.END, "• Ensure banana is clearly visible\n")
            self.results_text.insert(tk.END, "• Check image quality and lighting\n")

def main():
    root = tk.Tk()
    app = BananaAgeDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main()
