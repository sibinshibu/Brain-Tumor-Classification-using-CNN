# Brain-Tumor-Classification-using-CNN

# 🧠 Brain Tumor Classification Web App

This is a Streamlit web app to classify brain MRI images into four categories:
- `notumor`
- `glioma`
- `meningioma`
- `pituitary`

## 🔧 Features

- Upload MRI images (JPG, PNG)
- Select from pre-trained CNN models (`.h5`, `.keras`)
- Classify images with confidence scores
- Visualize predictions with a probability bar chart
- Show Grad-CAM heatmap to interpret model decisions

## 📁 Folder Structure
<pre><code> 📁 brain_tumor_app/ ├── 📁 models/ # Folder with your saved models │ ├── baseline_model.h5 │ ├── baseline_model.keras │ ├── fine_tune_model.h5 │ └── fine_tune_model.keras │ ├── 📄 app.py # Main Streamlit web app ├── 📄 requirements.txt # Dependencies for app └── 📄 README.md # Project documentation </code></pre>
