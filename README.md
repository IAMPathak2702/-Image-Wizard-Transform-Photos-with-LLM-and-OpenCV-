# Image-Wizard-Transform-Photos-with-LLM-and-OpenCV

Welcome to **Image Wizard**! This project leverages the power of Large Language Models (LLMs) to generate OpenCV commands from natural language descriptions. With an intuitive Streamlit interface, users can upload images and apply various transformations like brightness and contrast adjustment, blurring, and face detection. See the original and modified images side by side and experience the magic of AI-powered image processing.

## Key Features
- **Natural Language Command Conversion**: Convert your natural language commands into OpenCV code using LLMs.
- **Real-Time Image Manipulation**: Adjust brightness, contrast, and blur levels, and see the changes in real-time.
- **Face Detection**: Automatically detect and highlight faces in your images.
- **User-Friendly Interface**: Easily upload images and apply transformations using a clean Streamlit interface.
- **Side-by-Side Comparison**: View the original and modified images side by side for easy comparison.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/IAMPathak2702/Image-Wizard-Transform-Photos-with-LLM-and-OpenCV-.git
    cd image-wizard
    ```

2. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up OpenAI API key**:
    - Sign up at [OpenAI](https://beta.openai.com/signup/) if you haven't already.
    - Generate an API key from your OpenAI account dashboard.
    - Add your API key to the script by replacing `YOUR_OPENAI_API_KEY` with your actual OpenAI API key.

## Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2. **Interact with the app**:
    - Enter a natural language command (e.g., "Convert the image to grayscale").
    - Upload an image file.
    - Adjust the sliders in the sidebar for brightness, contrast, and blur, or check the "Detect Faces" option.
    - Click the button to generate and execute the OpenCV command.

## Example Commands

- "Increase the brightness of the image."
- "Apply a Gaussian blur to the image."
- "Detect faces in the image."
- "Convert the image to grayscale."

## Screenshots



## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

---

Dive in and explore how LLMs can simplify complex image processing tasks with **Image Wizard**!

