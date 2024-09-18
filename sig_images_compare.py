import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from scipy.spatial.distance import cosine
import numpy as np

# Define a simple Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)  # Output feature vector size 256

    def forward(self, x):
        return self.resnet(x)

# Load the model and set to evaluation mode
model = SiameseNetwork()
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to get image embeddings
def get_image_embedding(image, model, transform):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features

# Function to compare two images and return similarity
def compare_images(image1, image2, model, transform):
    embedding1 = get_image_embedding(image1, model, transform).squeeze().cpu().numpy()
    embedding2 = get_image_embedding(image2, model, transform).squeeze().cpu().numpy()
    
    # Calculate cosine similarity
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

# Streamlit UI
st.set_page_config(page_title="Signature Comparison App", layout="centered", initial_sidebar_state="expanded")

# App Title
st.title("🖋️ Signcheck AI Web App")
st.write("Upload two signature images to compare them and get a similarity score.")

# Image file uploaders
col1, col2 = st.columns(2)
with col1:
    st.subheader("Upload Signature 1")
    image1_file = st.file_uploader("Choose the first signature", type=["png", "jpg", "jpeg"])

with col2:
    st.subheader("Upload Signature 2")
    image2_file = st.file_uploader("Choose the second signature", type=["png", "jpg", "jpeg"])

# Placeholder for displaying the result
result_placeholder = st.empty()

if image1_file and image2_file:
    # Load images
    image1 = Image.open(image1_file).convert("RGB")
    image2 = Image.open(image2_file).convert("RGB")
    
    # Display images
    st.subheader("Uploaded Signatures:")
    col1, col2 = st.columns(2)
    col1.image(image1, caption="Signature 1", use_column_width=True)
    col2.image(image2, caption="Signature 2", use_column_width=True)
    
    # Compare signatures on button click
    if st.button("Compare Signatures"):
        with st.spinner("Comparing signatures..."):
            similarity_score = compare_images(image1, image2, model, transform)
        
        # Display similarity score
        result_placeholder.subheader(f"Similarity Score: {similarity_score:.2f}")
        
        # Interpret similarity
        if similarity_score > 0.98:
            st.success("The signatures are similar.")
        else:
            st.error("The signatures are different.")

st.markdown("---")
st.markdown("**Note:** No signature data is stored or saved. All images are immediately discareded after processing")

# Add a beautiful sidebar
st.sidebar.title("About This App")
st.sidebar.write("""
This app uses a Siamese Network powered by ResNet to compare two signature images. 
It calculates the cosine similarity between the two signature embeddings and provides a similarity score.
If the score is equals or above a certain threshold (0.98), the signatures are considered similar.
You can use this app for signature verification cutting across business lines.
""")

st.sidebar.markdown("### How It Works")
st.sidebar.write("""
- Upload two images of signatures.
- The app extracts features from both images using a deep learning model.
- It computes a similarity score based on the embeddings of the images.
""")

# Add footer with contact info or credits
st.sidebar.markdown("### Credits")
st.sidebar.write("Developed by [Kore Sampath Kumar].")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/kore-sampath-kumar-618b3762/)")
