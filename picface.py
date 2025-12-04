# ---------------------------------------
# Import the necessary packages
# ---------------------------------------
import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Face Detection",
    page_icon="👤",
    layout="centered"
)

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_image_features(image):
    """Detect faces, eyes, mouth in an uploaded image."""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Face
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_array, "Face", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img_array[y:y+h, x:x+w]

        # Eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            cv2.putText(roi_color, "Eye", (ex, ey - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # Mouth (using smile)
        mouth = smile_cascade.detectMultiScale(roi_gray, 1.4, 10)
        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0,165,255), 2)
            cv2.putText(roi_color, "Mouth", (mx, my - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
            break  # Only first detected mouth

    return img_array


# Streamlit App
def app():
    st.title("Face + Eyes + Mouth Detection (Image Upload)")
    st.write("Upload an image to detect features")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)

        processed = detect_image_features(image)
        st.image(processed, caption="Detection Result", use_column_width=True)


if __name__ == "__main__":
    app()
