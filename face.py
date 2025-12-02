# ---------------------------------------#
# Import the necessary packages
# ---------------------------------------#
import cv2
import streamlit as st
# Page configuration
st.set_page_config(
    page_title="Face Detection",
    page_icon="👤",
    layout="centered")
# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
# ---------------------------------------------------------------------------
#Function to Capture Webcam & Detect Faces
# ---------------------------------------------------------------------------
def detect_faces():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        #ret → True/False (did the camera successfully capture a frame?)
        #frame → the actual image captured from the webcam
        
        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect the faces
        #This loads a pre-trained model that knows how to detect faces
        faces = face_cascade.detectMultiScale( 
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        face_count = len(faces)

         # Display count at top-left corner
        cv2.putText(frame, f'Faces detected: {face_count}', 
           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
            cv2.putText(frame, 'Face', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close all windows
    cap.release() #Releases the webcam for other programs
    cv2.destroyAllWindows()
     # ---------------------------------------
    #streamlit app main function
 # ---------------------------------------


def app():
    st.title("Face Detection App")
    st.write("Click the button to start detecting faces!")
    
    if st.button("Start", type="primary"):
        detect_faces() # Call the detect_faces function

if __name__ == "__main__":
    app()