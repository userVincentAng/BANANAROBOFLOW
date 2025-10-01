import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import io
import base64

from utils import load_models, analyze_banana_ripeness, calculate_sugar_content, BananaSugarModel
from visualization import process_banana_detection
from storage import BananaStorage

# -----------------------------
# CSS Loading Function
# -----------------------------
def load_css(file_name):
    try:
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file {file_name} not found. Using default styles.")

# -----------------------------
# Drawer Sidebar Component
# -----------------------------
def create_drawer_sidebar():
    """Create a custom drawer sidebar with toggle control"""
    
    # Drawer toggle button (always visible)
    st.markdown("""
    <div class="sidebar-collapsed-control" onclick="toggleSidebar()">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M3 12h18M3 6h18M3 18h18"/>
        </svg>
    </div>
    """, unsafe_allow_html=True)
    
    # JavaScript for sidebar toggle
    st.markdown("""
    <script>
    function toggleSidebar() {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        const mainContent = document.querySelector('.main-content');
        
        if (sidebar) {
            const isHidden = sidebar.style.transform === 'translateX(-100%)';
            sidebar.style.transform = isHidden ? 'translateX(0)' : 'translateX(-100%)';
            sidebar.style.transition = 'transform 0.3s ease';
            
            if (mainContent) {
                mainContent.style.marginLeft = isHidden ? '0' : '0';
            }
        }
    }
    
    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', function(event) {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        const toggleBtn = document.querySelector('.sidebar-collapsed-control');
        
        if (window.innerWidth <= 768 && sidebar && toggleBtn && 
            !sidebar.contains(event.target) && !toggleBtn.contains(event.target)) {
            sidebar.style.transform = 'translateX(-100%)';
        }
    });
    
    // Handle window resize
    window.addEventListener('resize', function() {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (window.innerWidth > 768 && sidebar) {
            sidebar.style.transform = 'translateX(0)';
        }
    });
    </script>
    """, unsafe_allow_html=True)

# -----------------------------
# Responsive Header Component
# -----------------------------
def create_responsive_header():
    """Create responsive header with logos and school title"""
    st.markdown("""
    <div class="logo-container">
        <div class="logo-item">
    """, unsafe_allow_html=True)
    
    try:
        logo_image2 = Image.open("pcwhs.png")
        st.image(logo_image2, use_column_width=False)
    except Exception:
        st.warning("PCWHS logo not found")
    
    st.markdown("</div><div class='logo-item'>", unsafe_allow_html=True)
    
    try:
        logo_image1 = Image.open("pcwh_logo.png")
        st.image(logo_image1, use_column_width=False)
    except Exception:
        st.warning("PCWH logo not found")
    
    st.markdown("""
        </div>
        <div>
            <h2 class="school-title">Pasay City West High School — STE Program</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Streamlit UI Configuration
# -----------------------------
st.set_page_config(
    page_title="Lakatan Banana Ripeness & Sugar Content Detection", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS
load_css("styles.css")

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
    if 'has_peel' not in st.session_state:
        st.session_state.has_peel = True
    if 'sidebar_open' not in st.session_state:
        st.session_state.sidebar_open = False

# -----------------------------
# Sidebar Controls
# -----------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
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
        
        # PEEL TOGGLE - NEW FEATURE
        has_peel = st.checkbox(
            "🍌 Banana has peel", 
            value=st.session_state.has_peel,
            help="When checked, applies 0.6338 multiplier to sugar content (63.38% of sugar is in peeled banana)"
        )
        
        # Update session state when value changes
        if weight_per_banana != st.session_state.weight_per_banana:
            st.session_state.weight_per_banana = weight_per_banana
        if has_peel != st.session_state.has_peel:
            st.session_state.has_peel = has_peel
        
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
            - Peel multiplier: 0.6338 (63.38% of sugar in peeled banana)
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return sensitivity, show_spots, high_contrast_labels, weight_per_banana, sugar_formula, has_peel

# [Rest of the code remains the same as your previous version...]
# (Image Upload, Sugar Projection, Results Display, Storage Analytics, Footer, Main Application Flow)

# -----------------------------
# Main Application Flow
# -----------------------------
def main():
    initialize_session_state()
    
    # Create drawer sidebar toggle
    create_drawer_sidebar()
    
    # Main content wrapper
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Create responsive header
    create_responsive_header()
    
    st.title("🍌 Lakatan Banana Ripeness & Sugar Content Detection")
    
    # Load models once
    banana_model, coco_model = load_models()
    
    # Render sidebar
    sensitivity, show_spots, high_contrast_labels, weight_per_banana, sugar_formula, has_peel = render_sidebar()
    
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
                        image_bgr, banana_model, sensitivity, show_spots, high_contrast_labels, weight_per_banana, has_peel
                    )
                    
                    # Display annotated result
                    image_placeholder.image(results['image'], caption="Detection Result")
                    
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
                            sugar_content, formula_explanation, total_age, banana_count, sugar_breakdown, sugar_projections = calculate_sugar_content(
                                results['detections'], weight_per_banana, formula_key, has_peel
                            )
                        else:
                            sugar_content, formula_explanation, total_age, banana_count = calculate_sugar_content(
                                results['detections'], weight_per_banana, formula_key, has_peel
                            )
                            sugar_breakdown = None
                            sugar_projections = None
                        
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
                            sugar_content, formula_explanation, total_age, banana_count, 
                            sugar_breakdown, sugar_projections, has_peel
                        )
                        
                        # Display sugar projection if scientific model is used
                        if formula_key == "scientific" and sugar_projections:
                            display_sugar_projection(sugar_projections, results['detections'], weight_per_banana, has_peel)
                    else:
                        result_box.info("ℹ️ No bananas detected. Try adjusting sensitivity or use a clearer image.")
    
    with tab2:
        show_storage_analytics()
    
    render_footer()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()