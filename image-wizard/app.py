import openai
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Function to generate OpenCV command using OpenAI
def generate_opencv_command(natural_language_command, api_key):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Convert the following natural language command into OpenCV code: {natural_language_command}",
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Function to execute generated OpenCV command
def execute_opencv_command(opencv_command, image):
    local_namespace = {'cv2': cv2, 'np': np, 'image': image}
    exec(opencv_command, {}, local_namespace)
    return local_namespace.get('output_image', image)

# Function to adjust brightness and contrast
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    brightness = int((brightness - 50) * 2.55)
    contrast = int((contrast - 50) * 2.55)
    
    if brightness != 0:
        shadow = max(0, brightness)
        highlight = 255 if brightness > 0 else 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    
    return image

# Function to apply blur
def apply_blur(image, blur_level):
    return cv2.GaussianBlur(image, (2*blur_level + 1, 2*blur_level + 1), 0)

# Function to detect faces
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image, faces

# Function to adjust sharpness
def adjust_sharpness(image, sharpness):
    kernel = np.array([[0, -1, 0], [-1, 5 + sharpness, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Function to adjust hue
def adjust_hue(image, hue):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = hsv[..., 0] + hue
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Function to flip image
def flip_image(image, direction):
    if direction == 'Horizontal':
        return cv2.flip(image, 1)
    elif direction == 'Vertical':
        return cv2.flip(image, 0)
    else:
        return image

# Main function
def main():
    st.set_page_config(layout="wide")
    st.title("Natural Language to OpenCV with Image Manipulation")
    
    
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    
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
        detect_faces_option = st.sidebar.checkbox("Detect Faces")

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
        if detect_faces_option:
            modified_image, faces = detect_faces(modified_image)
            if faces:
                for i, (x, y, w, h) in enumerate(faces):
                    face_image = image[y:y+h, x:x+w]
                    face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
                    buf = BytesIO()
                    face_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(f"Download Face {i+1}", data=byte_im, file_name=f"face_{i+1}.png", mime="image/png")

        # Display images
        left , right = st.columns(2)
        
        with left:
            st.image(image, channels="BGR", caption="Original Image", use_column_width=True)
        
        with right:
            st.image(modified_image, channels="BGR", caption="Modified Image", use_column_width=True)

        # Generate and execute OpenCV command
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
        st.download_button("Download Modified Image", data=byte_im, file_name="modified_image.png", mime="image/png")

if __name__ == "__main__":
    main()
