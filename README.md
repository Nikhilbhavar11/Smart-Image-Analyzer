<h1 align="center">🔎 Smart Image Analyzer</h1>
<p align="center">
An AI-powered image analysis system built using Python, Streamlit, CLIP, BLIP, and Google Gemini API.
</p>

## 📌 Overview

The Smart Image Analyzer is an intelligent web application that automatically analyzes and explains images using advanced AI models.

It demonstrates the integration of multiple artificial intelligence technologies to convert visual information into meaningful insights.

Core concepts demonstrated in this project:

• Computer vision using CLIP<br>
• Image caption generation using BLIP<br>
• Contextual reasoning using Google Gemini<br>
• Knowledge retrieval using external APIs<br>
• Interactive web deployment using Streamlit<br>

The application allows users to upload an image and receive detailed explanations, key facts, and contextual information about the detected object.

---

## 🧩 Features

• Image Upload Interface<br>
• Drag-and-drop or browse image upload using Streamlit<br>

• AI Object Identification<br>
• CLIP (Contrastive Language–Image Pretraining) identifies the main object or scene category in the image<br>

• Automatic Image Captioning<br>
• BLIP (Bootstrapping Language Image Pretraining) generates descriptive captions for the uploaded image<br>

• Contextual AI Reasoning<br>
• Google Gemini API provides deeper explanations and structured insights about the object<br>

• Supplementary Knowledge Integration<br>
• Wikipedia API retrieves verified factual information<br>
• DuckDuckGo Instant Answer API provides additional contextual data<br>

• Structured Output Display<br>
• Main object identification<br>
• Detailed information about the object<br>
• Key facts and insights<br>
• Supplementary information section<br>

---

## 🛠 Tech Stack

• Programming Language: Python<br>
• Frontend Framework: Streamlit<br>
• Deep Learning Framework: PyTorch<br>
• Vision Models: OpenAI CLIP, Salesforce BLIP<br>
• Generative AI: Google Gemini API<br>
• Data APIs: Wikipedia API, DuckDuckGo Instant Answer API<br>
• Tools: VS Code / Python Environment, Git, Streamlit<br>

Libraries Used:

• Transformers<br>
• TorchVision<br>
• Pillow<br>
• Requests<br>

---

## ⚙️ Setup Instructions

**1️⃣ Clone the Repository**

git clone https://github.com/yourusername/smart-image-analyzer.git

cd smart-image-analyzer


**2️⃣ Install Dependencies**

pip install -r requirements.txt


**3️⃣ Configure Gemini API Key**

Add your Gemini API key in the project environment variables.

Example:

export GEMINI_API_KEY="your_api_key"


**4️⃣ Run the Application**

streamlit run main.py

---

## 🖼️ Screenshots

### 📤 Image Upload Interface
<p align="center">
  <img src="screenshots/upload_interface.png" width="700">
</p>

---

### 📷 Uploaded Image Preview
<p align="center">
  <img src="screenshots/uploaded_image.png" width="700">
</p>

---

### 🧠 Detailed AI Analysis
<p align="center">
  <img src="screenshots/detailed_analysis.png" width="700">
</p>

---

### 🌐 Supplementary Information
<p align="center">
  <img src="screenshots/supplementary_info.png" width="700">
</p>

---

## 📌 Future Improvements

• Multi-object detection using YOLOv8<br>
• Voice interaction (speech-to-text and text-to-speech)<br>
• Multilingual image explanations<br>
• Edge-device inference support<br>
• Mobile application deployment<br>
