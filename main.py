import os
import re
import requests
import wikipedia
import streamlit as st
from PIL import Image
import torch
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import BlipProcessor, BlipForConditionalGeneration
from google.genai import Client

# Load Models
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

clip_model, clip_preprocess, clip_device = load_clip()

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

blip_processor, blip_model = load_blip()

# Gemini
GENAI_API_KEY = ("AIzaSyBJHBhgeaQ3_TwDLPyOD9-EVYdkSGzUzZ0")
if not GENAI_API_KEY:
    st.error("❌ GENAI_API_KEY not set in environment variables.")
gemini_client = Client(api_key=GENAI_API_KEY) if GENAI_API_KEY else None

# Helper Functions
def detect_category(image: Image.Image) -> str:
    categories = ["landmark/building", "vehicle", "plant", "animal", "disease", "person", "object"]
    image_preprocessed = clip_preprocess(image).unsqueeze(0).to(clip_device)
    text_tokens = clip.tokenize(categories).to(clip_device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_preprocessed)
        text_features = clip_model.encode_text(text_tokens)
        logits = (image_features @ text_features.T).softmax(dim=-1)
    return categories[logits.argmax().item()]

def generate_blip_caption(image: Image.Image):
    img_resized = image.resize((384, 384))
    inputs = blip_processor(images=img_resized, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_length=50)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def clean_caption(caption: str) -> str:
    caption = re.sub(r'\(.*?\)', '', caption)
    caption = re.sub(r'^\s*(a|an|the)\s+(photo|picture|image|photograph)\s+of\s+', '', caption, flags=re.I)
    caption = re.split(r',| in the | with | at the | on the ', caption, flags=re.I)[0]
    return caption.strip().title()

def get_gemini_info(image, caption: str, category: str):
    if gemini_client is None:
        return "Gemini unavailable."

    prompt = f"""
Analyze the uploaded image and provide structured details in this exact format:

i. Main Object Identification: [Exact Name / Best Guess]
ii. Scientific Name: [If applicable / Unknown]
iii. Detailed Info:
   - Vehicles: company, brand, series, engine specs, history, features
   - Animals/Plants: species, habitat, features, conservation
   - Diseases: cause, symptoms, prevention, treatment
   - Landmarks/Buildings: location, architecture, history, cultural significance
   - People: notable identity, background, contributions
   - Objects: use, significance, context
iv. Key Facts: bullet points

Caption (if available): {caption or "N/A"}
Category hint: {category}
"""
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, prompt]
        )
        return resp.text
    except Exception as e:
        return f"Gemini call failed: {e}"

def get_wikipedia_info(query: str):
    try:
        return wikipedia.summary(query, sentences=5)
    except:
        return None

def get_duckduckgo_info(query: str):
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        res = requests.get(url).json()
        return res.get("AbstractText") or None
    except:
        return None

# Streamlit App
st.title("🔍 Smart Image Analyzer")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="content")

    # Step 1: Detect Category
    category = detect_category(image)

    # Step 2: BLIP (only for landmark/building)
    caption = None
    if category == "landmark/building":
        raw_caption = generate_blip_caption(image)
        caption = clean_caption(raw_caption)
        st.markdown(f"**📝 BLIP Caption:** {caption}")

    # Step 3: Gemini Info
    st.subheader("📑 Detailed Info about the Image")
    gemini_info = get_gemini_info(image, caption, category)
    st.text(gemini_info)

    # Step 4: Wikipedia + DuckDuckGo
    st.subheader("🌐 Supplementary Info")
    query = None
    for line in gemini_info.splitlines():
        if line.lower().startswith("i. main object identification"):
            query = line.split(":", 1)[1].strip()
            break

    if query:
        wiki = get_wikipedia_info(query)
        if wiki:
            st.markdown(f"**Wikipedia:** {wiki}")
        else:
            ddg = get_duckduckgo_info(query)
            st.markdown(f"**DuckDuckGo:** {ddg or 'No extra info found.'}")