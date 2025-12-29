import os
import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
from torchvision.transforms.functional import to_tensor
from kornia.feature import LoFTR
import tempfile
import zipfile
from io import BytesIO

st.set_page_config(
    page_title="RGB-Thermal Image Alignment",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4ECDC4;
    }
    </style>
""", unsafe_allow_html=True)

if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if 'loftr' not in st.session_state:
    with st.spinner("Loading LoFTR model..."):
        st.session_state.loftr = LoFTR(pretrained='outdoor').to(st.session_state.device).eval()

st.markdown('<p class="main-header">🔥 RGB-Thermal Image Alignment</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("### Image Settings")
    target_width = st.slider("Output Width", 320, 1920, 640, 32)
    target_height = st.slider("Output Height", 256, 1080, 512, 32)
    
    st.markdown("### Alignment Settings")
    invert_thermal = st.checkbox("Invert Thermal Image", value=True)
    min_keypoints = st.slider("Minimum Keypoints", 5, 50, 10, 1)
    
    st.markdown("---")
    st.markdown("### 🔗 Resources")
    st.markdown("[GitHub Repository](https://github.com/your-repo-link)")
    if torch.cuda.is_available():
        st.success("✅ GPU Available")
    else:
        st.warning("⚠️ CPU Mode")

def preprocess_for_loftr(image_bgr, target_size, invert=False):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = cv2.bitwise_not(gray)
    resized = cv2.resize(gray, target_size)
    tensor = to_tensor(resized.astype(np.float32) / 255.).unsqueeze(0).to(st.session_state.device)
    return tensor, resized

def align_images(thermal_img, rgb_img, target_size, invert_thermal, min_keypoints):
    thermal_resized = cv2.resize(thermal_img, target_size)
    rgb_resized = cv2.resize(rgb_img, target_size)
    
    thermal_tensor, _ = preprocess_for_loftr(thermal_resized, target_size, invert=invert_thermal)
    rgb_tensor, _ = preprocess_for_loftr(rgb_resized, target_size, invert=False)
    
    with torch.no_grad():
        output = st.session_state.loftr({'image0': thermal_tensor, 'image1': rgb_tensor})
    
    kpts0 = output['keypoints0'].cpu().numpy()
    kpts1 = output['keypoints1'].cpu().numpy()
    
    if len(kpts0) < min_keypoints:
        return None, f"Insufficient matches: {len(kpts0)} keypoints (minimum: {min_keypoints})"
    
    H, status = cv2.findHomography(kpts0, kpts1, cv2.RANSAC)
    if H is None:
        return None, "Homography estimation failed"
    
    aligned_thermal = cv2.warpPerspective(thermal_resized, H, target_size)
    
    return {
        'aligned_thermal': aligned_thermal,
        'rgb': rgb_resized,
        'num_keypoints': len(kpts0),
        'keypoints0': kpts0,
        'keypoints1': kpts1
    }, None

def create_comparison_image(rgb, aligned_thermal):
    return np.hstack([rgb, aligned_thermal])

def create_overlay_image(rgb, aligned_thermal, alpha=0.5):
    return cv2.addWeighted(rgb, alpha, aligned_thermal, 1-alpha, 0)

tab1, tab2 = st.tabs(["📤 Upload & Process", "📊 Results"])

with tab1:
    st.markdown("### Upload Image Pairs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌡️ Thermal Images")
        thermal_files = st.file_uploader(
            "Upload thermal images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="thermal"
        )
        if thermal_files:
            st.info(f"📁 {len(thermal_files)} thermal image(s) uploaded")
    
    with col2:
        st.markdown("#### 🎨 RGB Images")
        rgb_files = st.file_uploader(
            "Upload RGB images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="rgb"
        )
        if rgb_files:
            st.info(f"📁 {len(rgb_files)} RGB image(s) uploaded")
    
    st.markdown("---")
    
    if st.button("🚀 Process Images", type="primary", use_container_width=True):
        if not thermal_files or not rgb_files:
            st.error("⚠️ Please upload both thermal and RGB images!")
        elif len(thermal_files) != len(rgb_files):
            st.error(f"⚠️ Number of thermal ({len(thermal_files)}) and RGB ({len(rgb_files)}) images must match!")
        else:
            st.session_state.processed_images = []
            target_size = (target_width, target_height)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (thermal_file, rgb_file) in enumerate(zip(thermal_files, rgb_files)):
                status_text.text(f"Processing pair {idx+1}/{len(thermal_files)}: {thermal_file.name}")
                
                thermal_img = cv2.imdecode(np.frombuffer(thermal_file.read(), np.uint8), cv2.IMREAD_COLOR)
                rgb_img = cv2.imdecode(np.frombuffer(rgb_file.read(), np.uint8), cv2.IMREAD_COLOR)
                
                result, error = align_images(thermal_img, rgb_img, target_size, invert_thermal, min_keypoints)
                
                if error:
                    st.warning(f"⚠️ {thermal_file.name}: {error}")
                else:
                    comparison = create_comparison_image(result['rgb'], result['aligned_thermal'])
                    overlay = create_overlay_image(result['rgb'], result['aligned_thermal'])
                    
                    st.session_state.processed_images.append({
                        'thermal_name': thermal_file.name,
                        'rgb_name': rgb_file.name,
                        'rgb': result['rgb'],
                        'aligned_thermal': result['aligned_thermal'],
                        'comparison': comparison,
                        'overlay': overlay,
                        'num_keypoints': result['num_keypoints']
                    })
                
                progress_bar.progress((idx + 1) / len(thermal_files))
            
            status_text.empty()
            progress_bar.empty()
            
            if st.session_state.processed_images:
                st.success(f"✅ Successfully processed {len(st.session_state.processed_images)} image pair(s)!")
            else:
                st.error("❌ No images were successfully processed. Try adjusting the settings.")

with tab2:
    st.markdown("### 📊 Alignment Results")
    
    if not st.session_state.processed_images:
        st.info("👈 Upload and process images in the 'Upload & Process' tab to see results here.")
    else:
        image_idx = st.selectbox(
            "Select image pair",
            range(len(st.session_state.processed_images)),
            format_func=lambda x: f"Pair {x+1}: {st.session_state.processed_images[x]['thermal_name']}"
        )
        
        result = st.session_state.processed_images[image_idx]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Matched Keypoints", result['num_keypoints'])
        with col2:
            st.metric("Image Size", f"{target_width}×{target_height}")
        with col3:
            st.metric("Status", "✅ Success")
        
        st.markdown("---")
        
        viz_option = st.radio(
            "Visualization Mode",
            ["Side-by-Side", "Overlay", "Individual"],
            horizontal=True
        )
        
        if viz_option == "Side-by-Side":
            st.image(cv2.cvtColor(result['comparison'], cv2.COLOR_BGR2RGB), 
                    caption="RGB | Aligned Thermal", 
                    use_container_width=True)
        
        elif viz_option == "Overlay":
            alpha = st.slider("Overlay Alpha", 0.0, 1.0, 0.5, 0.1)
            overlay = create_overlay_image(result['rgb'], result['aligned_thermal'], alpha)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 
                    caption="RGB-Thermal Overlay", 
                    use_container_width=True)
        
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(result['rgb'], cv2.COLOR_BGR2RGB), 
                        caption="RGB", 
                        use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(result['aligned_thermal'], cv2.COLOR_BGR2RGB), 
                        caption="Aligned Thermal", 
                        use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 💾 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            _, buffer = cv2.imencode('.jpg', result['aligned_thermal'])
            st.download_button(
                label="Aligned Thermal",
                data=buffer.tobytes(),
                file_name=f"aligned_{result['thermal_name']}",
                mime="image/jpeg"
            )
        
        with col2:
            _, buffer = cv2.imencode('.jpg', result['comparison'])
            st.download_button(
                label="Comparison",
                data=buffer.tobytes(),
                file_name=f"comparison_{result['thermal_name']}",
                mime="image/jpeg"
            )
        
        with col3:
            _, buffer = cv2.imencode('.jpg', result['overlay'])
            st.download_button(
                label="Overlay",
                data=buffer.tobytes(),
                file_name=f"overlay_{result['thermal_name']}",
                mime="image/jpeg"
            )
        
        if len(st.session_state.processed_images) > 1:
            st.markdown("---")
            if st.button("📦 Download All as ZIP", use_container_width=True):
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for idx, img_result in enumerate(st.session_state.processed_images):
                        _, buffer = cv2.imencode('.jpg', img_result['aligned_thermal'])
                        zip_file.writestr(f"aligned_{idx+1}_{img_result['thermal_name']}", buffer.tobytes())
                        
                        _, buffer = cv2.imencode('.jpg', img_result['comparison'])
                        zip_file.writestr(f"comparison_{idx+1}_{img_result['thermal_name']}", buffer.tobytes())
                
                st.download_button(
                    label="⬇️ Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="aligned_results.zip",
                    mime="application/zip"
                )
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Built with Streamlit
</div>
""", unsafe_allow_html=True)