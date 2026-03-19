import streamlit as st
from PIL import Image
import numpy as np
import io
import rembg
from streamlit_drawable_canvas import st_canvas
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Configure Page
st.set_page_config(page_title="Premium Background Remover", page_icon="🪄", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: transform 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        color: white;
    }
    .stDownloadButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        border: none;
        padding: 0.75rem 1rem;
        font-weight: bold;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .gradient-text {
        background: linear-gradient(135deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# Define available models
AVAILABLE_MODELS = {
    "ISNet (Ultimate Quality General)": "isnet-general-use",
    "ISNet Anime (Best for Cartoons)": "isnet-anime",
    "U2Net (Fast General)": "u2net",
    "U2Net Human (Focus on Portraits)": "u2net_human_seg"
}

@st.cache_resource
def load_single_model(model_name):
    # Only load the exact rembg model requested
    return rembg.new_session(model_name)

@st.cache_resource
def load_clipseg():
    # Only load the heavy CLIPSeg model when requested
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    return processor, model


# --- HEADER ---
st.markdown("<div class='main-header'><h1>AI Background <span class='gradient-text'>Studio</span></h1><p>High-quality, intelligent foreground extraction with Manual Touch-ups.</p></div>", unsafe_allow_html=True)

# --- SIDEBAR TUNERS ---
with st.sidebar:
    st.header("⚙️ AI Settings")
    model_choice = st.selectbox("AI Model", options=list(AVAILABLE_MODELS.keys()))
    
    st.divider()
    st.header("🎛️ Accuracy Tuners")
    st.caption("Adjust edge detection sensitivity.")
    enable_alpha = st.toggle("Enable Advanced Edge Smoothing", value=True)
    
    if enable_alpha:
        fg_thresh = st.slider("Foreground Threshold", min_value=0, max_value=255, value=240, help="Higher = stricter foreground detection")
        bg_thresh = st.slider("Background Threshold", min_value=0, max_value=255, value=10, help="Lower = stricter background removal")
        erode_size = st.slider("Erode Size", min_value=0, max_value=50, value=10, help="Smooths out the jagged edges")
    else:
        fg_thresh, bg_thresh, erode_size = 240, 10, 10

    st.divider()
    st.header("✍️ Text Prompt Selection")
    st.caption("Optional: Type what you want to extract (e.g., 'dog'). *Note: Experimental, works best on simple objects.* Leave empty for automatic extraction.")
    text_prompt = st.text_input("Object to Extract:", value="")
    if text_prompt:
        mask_threshold = st.slider("Text Mask Strictness", min_value=0.0, max_value=1.0, value=0.4, help="Higher = stricter selection of the text object")


# --- MAIN WORKSPACE ---
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGBA")
    orig_w, orig_h = original_image.size
    
    # Check max dimensions
    MAX_DIM = 4096
    if orig_w > MAX_DIM or orig_h > MAX_DIM:
        st.warning(f"Image is massive ({orig_w}x{orig_h}). Scaling down to {MAX_DIM}px for safety.")
        scale = min(MAX_DIM / orig_w, MAX_DIM / orig_h)
        original_image = original_image.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
        orig_w, orig_h = original_image.size

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)

    with col2:
        st.subheader("AI Preview")
        
        # --- PHASE 2: AI PROCESSING ---
        with st.spinner("🤖 Analyzing pixels..."):
            
            # Safe downscaling for neural network to avoid RAM crash
            proc_dim = 1024
            scale = min(proc_dim / orig_w, proc_dim / orig_h)
            if scale < 1.0:
                proc_image = original_image.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
            else:
                proc_image = original_image

            # AI Check: Text Prompt vs Standard Rembg
            if text_prompt.strip():
                # --- PHASE 2b: CLIPSEG TEXT-PROMPTED MASK ---
                prompt_text = text_prompt.strip()
                try:
                    clipseg_processor, clipseg_model = load_clipseg()
                    
                    inputs = clipseg_processor(
                        text=[prompt_text], images=[proc_image.convert("RGB")], padding="max_length", return_tensors="pt"
                    )
                    
                    with torch.no_grad():
                        outputs = clipseg_model(**inputs)
                    
                    preds = outputs.logits.unsqueeze(1)
                    
                    # Apply sigmoid and threshold
                    pred_mask = torch.sigmoid(preds[0][0])
                    pred_mask = (pred_mask > mask_threshold).float().numpy()
                    
                    # Convert the numerical mask back to a standard PIL Mask (0 to 255)
                    pred_mask = (pred_mask * 255).astype(np.uint8)
                    ai_mask = Image.fromarray(pred_mask, mode="L")
                except Exception as e:
                    st.error(f"Failed to load Text AI Model. Error: {e}")
                    st.stop()
            else:
                # --- PHASE 2a: STANDARD REMBG MASK ---
                try:
                    # Lazy load only the specific model requested
                    active_session = load_single_model(AVAILABLE_MODELS[model_choice])
                except Exception as e:
                    st.error(f"Failed to load standard AI Model. Error: {e}")
                    st.stop()
                    
                kwargs = {
                    "session": active_session,
                    "only_mask": True,
                    "alpha_matting": enable_alpha,
                    "alpha_matting_foreground_threshold": fg_thresh,
                    "alpha_matting_background_threshold": bg_thresh,
                    "alpha_matting_erode_size": erode_size,
                    "post_process_mask": True
                }
                ai_mask = rembg.remove(proc_image, **kwargs)
            
            # Upscale mask safely back to original bounds using BILINEAR
            if ai_mask.size != (orig_w, orig_h):
                ai_mask = ai_mask.resize((orig_w, orig_h), Image.BILINEAR)
            
            if ai_mask.mode != "L":
                ai_mask = ai_mask.convert("L")

        # Create live composite for AI preview
        ai_result = original_image.copy()
        ai_result.putalpha(ai_mask)
        st.image(ai_result, use_container_width=True)

        # Immediate AI-only Preview Download
        buf_ai = io.BytesIO()
        ai_result.save(buf_ai, format="PNG")
        byte_im_ai = buf_ai.getvalue()
        st.download_button(
            label="⬇️ Download AI Result",
            data=byte_im_ai,
            file_name="ai_remove_bg.png",
            mime="image/png",
            key="dl_ai" # Add key to prevent id conflicts
        )
