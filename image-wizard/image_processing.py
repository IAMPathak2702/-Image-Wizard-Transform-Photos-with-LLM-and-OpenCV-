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
        dimensions = (x,y , x+w , y+h)
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


# Function to rotate image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


# Function to apply style transfer
def apply_style_transfer(content_image, style_image_path):
    style_image = Image.open(style_image_path)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    content_tensor = transform(content_image).unsqueeze(0)
    style_tensor = transform(style_image).unsqueeze(0)

    style_model = vgg19(pretrained=True).features.eval()
    content_features = get_features(content_tensor, style_model)
    style_features = get_features(style_tensor, style_model)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    target = content_tensor.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([target], lr=0.003)
    style_weight = 1e6
    content_weight = 1e0

    for i in range(300):
        target_features = get_features(target, style_model)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        style_loss = 0
        for layer in style_grams:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]
            style_loss += torch.mean((target_gram - style_gram)**2) / (d * h * w)
        total_loss = style_weight * style_loss + content_weight * content_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    target = target.squeeze().detach().cpu()
    target_image = transforms.ToPILImage()(target)
    return target_image

def get_features(image, model):
    layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Function to colorize image
def colorize_image(image):
    colorization_model = segmentation.deeplabv3_resnet50(pretrained=True).eval()
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = colorization_model(input_tensor)['out'][0]
    output_predictions = output.argmax(0)
    return transforms.ToPILImage()(output_predictions.byte())

# Function to enhance image resolution
def super_resolve_image(image):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    output = torch.nn.functional.interpolate(output, scale_factor=4, mode='bicubic')
    return transforms.ToPILImage()(output.squeeze())

# Function to inpaint image
def inpaint_image(image, mask):
    inpainting_model = deeplabv3_resnet50(pretrained=True).eval()
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)
    mask_tensor = transform(mask).unsqueeze(0)
    with torch.no_grad():
        output = inpainting_model(input_tensor * mask_tensor)['out'][0]
    output_predictions = output.argmax(0)
    return transforms.ToPILImage()(output_predictions.byte())