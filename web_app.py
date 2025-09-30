import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import io

from utils import load_models, analyze_banana_ripeness, calculate_sugar_content, BananaSugarModel
from visualization import process_banana_detection
from storage import BananaStorage

# -----------------------------
# Streamlit UI Configuration
# -----------------------------
st.set_page_config(page_title="Lakatan Ripeness & Sugar Content Detection", layout="wide")
st.title("🍌 Lakatan Ripeness & Sugar Content Detection")

# Initialize storage
@st.cache_resource
def init_storage():
    return BananaStorage()

storage = init_storage()

# -----------------------------
# Session State Initialization
# -----------------------------
def initialize_session_state():
    if 'weight_per_banana' not in st.session_state:
        st.session_state.weight_per_banana = 120.0
    if 'sugar_formula' not in st.session_state:
        st.session_state.sugar_formula = "Scientific: Regression Model (R²=0.87)"

# -----------------------------
# Sidebar Controls
# -----------------------------
def render_sidebar():
    with st.sidebar:
        st.header("🎛️ Controls")
        sensitivity = st.slider("Detection Sensitivity", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
        show_spots = st.checkbox("Show Spots", value=True)
        high_contrast_labels = st.checkbox("High Contrast Labels", value=True)
        
        st.subheader("🔬 Scientific Analysis")
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
        
        # Scientific formula selection
        sugar_formula = st.selectbox(
            "Sugar Prediction Model",
            options=[
                "Scientific: Regression Model (R²=0.87)",
                "Basic: (Total Age × Weight) / 100", 
                "Advanced: (Total Age × Weight × Ripeness Factor) / 100"
            ],
            index=0 if st.session_state.sugar_formula.startswith("Scientific") else 1,
            help="Scientific model based on experimental data: Sugar% = 25.99 + 0.707 × Age"
        )
        if sugar_formula != st.session_state.sugar_formula:
            st.session_state.sugar_formula = sugar_formula

        # Storage Management Section
        st.subheader("💾 Storage Management")
        
        # Show storage stats
        stats = storage.get_storage_stats()
        st.info(f"📊 Storage: {stats['images']}/10 images, {stats['json_analyses']}/10 analyses")
        
        if st.button("🔄 Clear All Storage"):
            storage.clear_storage()
            st.rerun()
        
        if st.button("📤 Export Data"):
            export_path = storage.export_data()
            st.success(f"Data exported to: {export_path}")

        # Individual analysis management
        with st.expander("🗑️ Manage Individual Analyses"):
            analyses = storage.get_all_analyses_with_ids()
            if analyses:
                for analysis in analyses:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"**{analysis['timestamp'][:16]}**")
                        st.write(f"{analysis['banana_count']} bananas, {analysis['total_sugar_grams']:.1f}g sugar")
                    with col_b:
                        if st.button("🗑️", key=f"delete_{analysis['id']}", help="Delete this analysis"):
                            storage.delete_analysis_by_id(analysis['id'])
                            st.rerun()
            else:
                st.info("No analyses to manage")
        
        # Quick delete options
        with st.expander("⚡ Quick Actions"):
            if st.button("🗑️ Delete Oldest Analysis"):
                if storage.delete_oldest_analysis():
                    st.success("Oldest analysis deleted!")
                    st.rerun()
                else:
                    st.info("No analyses to delete")
            
            if st.button("📷 Clear Images Only"):
                # Clear only images, keep analysis data
                image_files = list(storage.images_dir.glob("*.jpg"))
                for image_file in image_files:
                    image_file.unlink()
                st.success(f"Cleared {len(image_files)} images!")
                st.rerun()
        
        # Show recent analyses
        recent = storage.get_recent_analyses(3)
        if recent:
            with st.expander("📝 Recent Analyses"):
                for analysis in recent:
                    st.write(f"**{analysis['timestamp'][:16]}** - {analysis['banana_count']} bananas, {analysis['total_sugar_grams']:.1f}g sugar")

        # Show dataset info
        with st.expander("📊 About the Scientific Model"):
            st.markdown("""
            **Dataset Information:**
            - Based on experimental measurements
            - Regression: Sugar% = 25.99 + 0.707 × Age
            - R-squared: 0.867 (High correlation)
            - Age range: 1-8 days
            - Measurements taken at stem, middle, and tip
            """)
    
    return sensitivity, show_spots, high_contrast_labels, weight_per_banana, sugar_formula

# -----------------------------
# Image Upload & Display
# -----------------------------
def handle_image_upload():
    uploaded_file = st.file_uploader("📤 Upload an image of bananas", type=["jpg", "jpeg", "png", "bmp"])
    image_bgr = None
    
    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image_bgr is not None:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                return image_bgr, image_rgb, uploaded_file.name
        except Exception as e:
            st.error(f"❌ Failed to load image: {e}")
    
    return None, None, None

# -----------------------------
# Results Display (UPDATED FUNCTION SIGNATURE)
# -----------------------------
def display_detection_results(results, formula_type, weight_per_banana, result_box, 
                             sugar_content, formula_explanation, total_age, banana_count, sugar_breakdown=None):
    """Updated function to accept all required parameters"""
    if not results.get('detections'):
        result_box.info("ℹ️ No bananas detected. Try adjusting sensitivity or use a clearer image.")
        return
    
    # Enhanced results display
    lines = [
        "🔬 SCIENTIFIC BANANA ANALYSIS", 
        "=" * 50, 
        ""
    ]
    
    # Determine formula type for display
    formula_type_map = {
        "Scientific: Regression Model (R²=0.87)": "scientific",
        "Basic: (Total Age × Weight) / 100": "basic",
        "Advanced: (Total Age × Weight × Ripeness Factor) / 100": "advanced"
    }
    formula_key = formula_type_map[formula_type]
    
    # Model information for scientific approach
    if formula_key == "scientific":
        lines.extend([
            "📊 REGRESSION MODEL (From Experimental Data)",
            f"• Formula: Sugar% = 25.99 + 0.707 × Age",
            f"• R-squared: 0.867 (High correlation)",
            f"• Standard Error: ±0.65%",
            ""
        ])
    
    # Nutritional summary
    lines.extend([
        "📈 ANALYSIS SUMMARY",
        f"• Bananas Detected: {banana_count}",
        f"• Total Age: {total_age:.1f} days",
        f"• Average Weight per Banana: {weight_per_banana:.1f}g",
        f"• Total Weight: {weight_per_banana * banana_count:.1f}g",
        f"• Estimated Total Sugar Content: {sugar_content:.1f}g",
        f"• Calculation Method: {formula_explanation}",
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
    lines.append("🍌 INDIVIDUAL BANANA ANALYSIS")
    for i, det in enumerate(results['detections']):
        a = det['analysis']
        
        if formula_key == "scientific":
            sugar_info = f"{det['sugar_percentage']:.1f}% | {det['sugar_content']:.1f}g"
        else:
            sugar_info = f"{det['sugar_content']:.1f}g"
            
        lines.extend([
            f"Banana {i + 1} Results:",
            f"  • Age: {a['age_days']} days",
            f"  • Sugar Content: {sugar_info}",
            f"  • Ripeness Score: {a['ripeness_score']}/5.0",
            f"  • Stage: {det['ripeness_stage']}",
            f"  • Color Analysis - Green: {a['green_percent']:.1f}%, Yellow: {a['yellow_percent']:.1f}%, Brown: {a['brown_percent']:.1f}%",
            f"  • Spot Coverage: {a['spot_percent']:.1f}% ({a['spot_count']} spots)",
            ""
        ])
    
    result_box.code("\n".join(lines))
    display_visualizations(formula_key, sugar_content, sugar_breakdown, results['detections'])

def display_visualizations(formula_key, sugar_content, sugar_breakdown, detections):
    with st.expander("📈 Scientific Visualizations", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Sugar progression chart
            if formula_key == "scientific":
                sugar_model = BananaSugarModel()
                ages = np.arange(0, 11, 1)
                sugar_percentages = [sugar_model.predict_sugar_percentage(age) for age in ages]
                
                st.write("**Sugar vs Age Progression**")
                chart_data = pd.DataFrame({
                    'Age (days)': ages,
                    'Sugar %': sugar_percentages
                })
                st.line_chart(chart_data.set_index('Age (days)'))
        
        with col2:
            # Model accuracy metrics
            if formula_key == "scientific":
                st.metric("Model R-squared", "0.867")
                st.metric("Correlation Strength", "Strong")
                st.metric("Standard Error", "±0.65%")
        
        with col3:
            # Daily intake context
            daily_percentage = (sugar_content / 25) * 100
            st.metric("Daily Sugar Intake", f"{daily_percentage:.1f}%")
            st.progress(min(100, daily_percentage)/100)
            st.metric("Teaspoons Equivalent", f"{sugar_content / 4:.1f}")
    
    # Show detailed sugar breakdown for scientific model
    if formula_key == "scientific" and sugar_breakdown:
        with st.expander("🔍 Detailed Sugar Breakdown"):
            breakdown_data = []
            for item in sugar_breakdown:
                breakdown_data.append({
                    'Banana': item['banana'],
                    'Age (days)': item['age_days'],
                    'Sugar Percentage': f"{item['sugar_percentage']:.1f}%",
                    'Sugar Content (g)': f"{item['sugar_grams']:.1f}g"
                })
            st.table(breakdown_data)
    
    # Ripeness distribution
    with st.expander("🍌 Ripeness Distribution"):
        if detections:
            ripeness_scores = [det['analysis']['ripeness_score'] for det in detections]
            stages = [det['ripeness_stage'] for det in detections]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Ripeness Scores**")
                score_data = pd.DataFrame({
                    'Banana': range(1, len(ripeness_scores) + 1),
                    'Ripeness Score': ripeness_scores
                })
                st.bar_chart(score_data.set_index('Banana'))
            
            with col2:
                st.write("**Age Distribution**")
                ages = [det['analysis']['age_days'] for det in detections]
                age_data = pd.DataFrame({
                    'Banana': range(1, len(ages) + 1),
                    'Age (days)': ages
                })
                st.bar_chart(age_data.set_index('Banana'))

def show_storage_analytics():
    st.subheader("📈 Storage Analytics")
    
    # Storage overview
    col1, col2, col3 = st.columns(3)
    stats = storage.get_storage_stats()
    
    with col1:
        st.metric("Stored Images", f"{stats['images']}/10", 
                 help="Number of stored images out of maximum 10")
    
    with col2:
        st.metric("JSON Analyses", f"{stats['json_analyses']}/10",
                 help="Number of analysis records in JSON file")
    
    with col3:
        st.metric("Database Records", f"{stats['db_analyses']}/10",
                 help="Number of records in SQLite database")
    
    # Storage actions
    st.subheader("🛠️ Storage Actions")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🗑️ Clear All Storage", key="clear_all_storage_analytics"):
            storage.clear_storage()
            st.rerun()
    
    with action_col2:
        if st.button("📤 Export All Data", key="export_all_data_analytics"):
            export_path = storage.export_data()
            st.success(f"✅ Data exported to: `{export_path}`")
    
    with action_col3:
        if st.button("🔄 Delete Oldest", key="delete_oldest_analytics"):
            if storage.delete_oldest_analysis():
                st.success("Oldest analysis deleted!")
                st.rerun()
            else:
                st.info("No analyses to delete")
    
    # Detailed analysis management
    st.subheader("📋 Analysis Management")
    
    analyses = storage.get_all_analyses_with_ids()
    if analyses:
        # Create a dataframe for better display
        df_data = []
        for analysis in analyses:
            df_data.append({
                'ID': analysis['id'],
                'Timestamp': analysis['timestamp'][:16],
                'Bananas': analysis['banana_count'],
                'Total Sugar (g)': analysis['total_sugar_grams'],
                'Avg Ripeness': analysis['avg_ripeness_score'],
                'Image File': analysis['image_filename']
            })
        
        df = pd.DataFrame(df_data)
        
        # Display with delete options
        for index, row in df.iterrows():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.write(f"**{row['Timestamp']}**")
                st.write(f"Image: `{row['Image File']}`")
            
            with col2:
                st.write(f"🍌 {row['Bananas']} bananas")
                st.write(f"📊 {row['Avg Ripeness']:.1f}/5.0 ripeness")
            
            with col3:
                st.write(f"🍬 {row['Total Sugar (g)']:.1f}g sugar")
                st.write(f"🆔 ID: {row['ID']}")
            
            with col4:
                if st.button("🗑️ Delete", key=f"delete_analytics_{row['ID']}"):
                    storage.delete_analysis_by_id(row['ID'])
                    st.rerun()
            
            st.divider()
        
        # Bulk actions
        st.subheader("🎯 Bulk Actions")
        bulk_col1, bulk_col2 = st.columns(2)
        
        with bulk_col1:
            if st.button("🗑️ Delete All Analyses", key="delete_all_analyses"):
                storage.clear_storage()
                st.rerun()
        
        with bulk_col2:
            if st.button("📷 Clear Images Only", key="clear_images_only"):
                image_files = list(storage.images_dir.glob("*.jpg"))
                for image_file in image_files:
                    image_file.unlink()
                st.success(f"Cleared {len(image_files)} images!")
                st.rerun()
    
    else:
        st.info("📭 No analyses stored yet. Upload and analyze some bananas to see data here!")
    
    # Storage usage visualization
    st.subheader("📊 Storage Usage")
    
    if analyses:
        # Create usage chart
        usage_data = {
            'Type': ['Images', 'JSON Analyses', 'DB Records'],
            'Used': [stats['images'], stats['json_analyses'], stats['db_analyses']],
            'Total': [10, 10, 10]
        }
        
        usage_df = pd.DataFrame(usage_data)
        usage_df['Percentage'] = (usage_df['Used'] / usage_df['Total']) * 100
        
        # Display progress bars
        for _, row in usage_df.iterrows():
            st.write(f"**{row['Type']}**: {row['Used']}/{row['Total']}")
            st.progress(row['Percentage'] / 100)
            st.write("")

# -----------------------------
# Footer
# -----------------------------
def render_footer():
    st.markdown("---")
    st.markdown("""
    **🔬 Scientific Basis:**
    - Sugar prediction based on experimental data with R²=0.867
    - Regression formula: Sugar% = 25.99 + 0.707 × Age
    - Measurements validated across banana stem, middle, and tip
    - Age range: 1-8 days with extrapolation to 10 days
    
    **💡 Tips for best results:**
    - Use clear, well-lit images of bananas
    - Ensure bananas are visible and not overlapping too much
    - Adjust sensitivity if detection is too strict/lenient
    - Typical banana weight is 100-150g for accurate sugar estimation
    """)

# -----------------------------
# Main Application Flow
# -----------------------------
def main():
    initialize_session_state()
    
    # Load models once
    banana_model, coco_model = load_models()
    
    # Render sidebar
    sensitivity, show_spots, high_contrast_labels, weight_per_banana, sugar_formula = render_sidebar()
    
    # Add storage analytics tab
    tab1, tab2 = st.tabs(["🔍 Banana Analysis", "📊 Storage Analytics"])
    
    with tab1:
        # Layout columns
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            image_placeholder = st.empty()
        
        with col_right:
            st.subheader("📊 Detection Results")
            result_box = st.empty()
        
        # Handle image upload
        image_bgr, image_rgb, filename = handle_image_upload()
        if image_rgb is not None:
            image_placeholder.image(image_rgb, caption=f"Uploaded Image: {filename}")
        
        # Detection button
        detect_clicked = st.button("🔍 Detect Ripeness & Sugar Content", type="primary")
        
        # Process detection
        if detect_clicked:
            if image_bgr is None:
                st.warning("⚠️ Please upload an image first.")
            else:
                with st.spinner("🔬 Analyzing bananas with scientific model..."):
                    # Process banana detection
                    results = process_banana_detection(
                        image_bgr, banana_model, sensitivity, show_spots, high_contrast_labels, weight_per_banana
                    )
                    
                    # Display annotated result
                    image_placeholder.image(results['image'], caption="Detection Result",  )
                    
                    if results.get('detections'):
                        # Determine formula type
                        formula_type_map = {
                            "Scientific: Regression Model (R²=0.87)": "scientific",
                            "Basic: (Total Age × Weight) / 100": "basic",
                            "Advanced: (Total Age × Weight × Ripeness Factor) / 100": "advanced"
                        }
                        formula_key = formula_type_map[sugar_formula]
                        
                        # Calculate sugar content
                        if formula_key == "scientific":
                            sugar_content, formula_explanation, total_age, banana_count, sugar_breakdown = calculate_sugar_content(
                                results['detections'], weight_per_banana, formula_key
                            )
                        else:
                            sugar_content, formula_explanation, total_age, banana_count = calculate_sugar_content(
                                results['detections'], weight_per_banana, formula_key
                            )
                            sugar_breakdown = None
                        
                        # Save to storage
                        try:
                            # Convert image to bytes for storage
                            img_bytes = io.BytesIO()
                            results['image'].save(img_bytes, format='JPEG')
                            img_bytes = img_bytes.getvalue()
                            
                            # Generate unique filename
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"banana_analysis_{timestamp}.jpg"
                            
                            # Save image and analysis
                            saved_filename = storage.save_image(img_bytes, filename)
                            analysis_data = storage.save_analysis(
                                saved_filename, 
                                results['detections'], 
                                sugar_content, 
                                sugar_formula, 
                                weight_per_banana
                            )
                            
                            st.success(f"✅ Analysis saved to storage ({storage.get_storage_stats()['json_analyses']}/10 analyses)")
                            
                        except Exception as e:
                            st.error(f"❌ Failed to save analysis: {e}")
                        
                        # Display results with all required parameters
                        display_detection_results(
                            results, sugar_formula, weight_per_banana, result_box,
                            sugar_content, formula_explanation, total_age, banana_count, sugar_breakdown
                        )
                    else:
                        result_box.info("ℹ️ No bananas detected. Try adjusting sensitivity or use a clearer image.")
    
    with tab2:
        show_storage_analytics()
    
    render_footer()

if __name__ == "__main__":
    main()