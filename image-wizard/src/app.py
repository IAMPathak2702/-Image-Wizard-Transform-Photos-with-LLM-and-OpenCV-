import openai
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
from torchvision.models import vgg19, segmentation, resnet50
from torchvision.models.segmentation import deeplabv3_resnet50
from ultralytics import YOLO

# Function to execute generated OpenCV command

from image_processing import apply_blur,adjust_brightness_contrast,adjust_sharpness
from image_processing import execute_opencv_command, detect_faces , adjust_hue
from image_processing import flip_image, rotate_image, execute_opencv_command
from image_processing import resnet50,rotate_image,super_resolve_image, apply_style_transfer
from image_processing import get_features , gram_matrix , colorize_image
from image_processing import deeplabv3_resnet50 , super_resolve_image , inpaint_image




# Main function
def main():
    st.set_page_config(layout="wide")
    st.title("Natural Language to OpenCV with Image Manipulation")
    
    
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    if api_key:
        natural_language_command = st.chat_input("Enter a natural language command:")
        
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Sidebar for image manipulation
        st.sidebar.title("Image Manipulation")
        brightness = st.sidebar.slider("Brightness", 0, 100, 50)
        contrast = st.sidebar.slider("Contrast", 0, 100, 50)
        blur_level = st.sidebar.slider("Blur", 0, 10, 0)
        sharpness = st.sidebar.slider("Sharpness", 0, 10, 0)
        hue = st.sidebar.slider("Hue", 0, 180, 0)
        flip_direction = st.sidebar.selectbox("Flip Image", ["None", "Horizontal", "Vertical"])
        rotation_angle = st.sidebar.slider("Rotate Image", -180, 180, 0)
        detect_faces_option = st.sidebar.checkbox("Detect Faces")
        style_transfer_option = st.sidebar.checkbox("Apply Style Transfer")
        colorize_option = st.sidebar.checkbox("Colorize Image")
        super_res_option = st.sidebar.checkbox("Super Resolution")
        inpainting_option = st.sidebar.checkbox("Inpaint Image")
        inpainting_mask = None

        if inpainting_option:
            inpainting_mask_file = st.file_uploader("Upload an inpainting mask", type=["png"])
            if inpainting_mask_file is not None:
                inpainting_mask = Image.open(inpainting_mask_file)
        
        # Apply image manipulations
        modified_image = adjust_brightness_contrast(image, brightness, contrast)
        if blur_level > 0:
            modified_image = apply_blur(modified_image, blur_level)
        if sharpness > 0:
            modified_image = adjust_sharpness(modified_image, sharpness)
        if hue != 0:
            modified_image = adjust_hue(modified_image, hue)
        if flip_direction != "None":
            modified_image = flip_image(modified_image, flip_direction)
        if rotation_angle != 0:
            modified_image = rotate_image(modified_image, rotation_angle)
        if detect_faces_option:
            modified_image, faces = detect_faces(modified_image)
            if faces is not None and faces.size > 0:  
                for i, (x, y, w, h) in enumerate(faces):
                    face_image = image[y:y+h, x:x+w]
                    face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
                    buf = BytesIO()
                    face_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(f"Download Face {i+1}", data=byte_im, file_name=f"face_{i+1}.png", mime="image/png")

        
        # Convert modified_image back to PIL for further manipulations
        modified_image_pil = Image.fromarray(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))

        if style_transfer_option:
            style_image_file = st.file_uploader(label="style image", type=["jpeg", "jpg","png"])
            if style_image_file:
                modified_image_pil = apply_style_transfer(modified_image_pil, style_image_file)

        if colorize_option:
            modified_image_pil = colorize_image(modified_image_pil)

        if super_res_option:
            modified_image_pil = super_resolve_image(modified_image_pil)

        if inpainting_option and inpainting_mask:
            modified_image_pil = inpaint_image(modified_image_pil, inpainting_mask)
        
        # Display images
        left , right = st.columns(2)
        
        with left:
            st.image(image, channels="BGR", caption="Original Image", use_column_width=True)
        
        with right:
            st.image(modified_image, channels="BGR", caption="Modified Image", use_column_width=True)

        # Generate and execute OpenCV command
        l1 , r1 = st.columns(2)
        with l1:
            if st.button("Generate and Execute OpenCV Command") and api_key:
                opencv_command = generate_opencv_command(natural_language_command, api_key)
                st.text_area("Generated OpenCV Command", opencv_command)
                
                try:
                    output_image = execute_opencv_command(opencv_command, image)
                    st.image(output_image, channels="BGR", caption="Output Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error executing command: {e}")

        # Provide download button for modified image
        modified_pil = Image.fromarray(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        modified_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        with r1:
            st.download_button("Download Modified Image", data=byte_im, file_name="modified_image.png", mime="image/png")

if __name__ == "__main__":
    main()
