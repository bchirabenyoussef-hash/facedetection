# ---------------------------------------
# Import the necessary packages
# ---------------------------------------
import cv2
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Face Detection",
    page_icon="👤",
    layout="centered")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
def detect_faces():
    cap = cv2.VideoCapture(0) #open the default camera /0 → built-in webcam
    while True:
        ret, frame = cap.read()# read the frames from the webcam
        #ret → True/False (did we get an image?)
        #frame → the actual image from the webcam.
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert to grayscale
        #AI detection works better on black & white images.
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # Draw face box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        #----------------------------------------------------------------------------
# FEATURE DETECTION INSIDE FACES
#----------------------------------------------------------------------------
            # defone the ROI Region of Interest
            #we only search inside the face area → faster & more accurate

            roi_gray = gray[y:y+h, x:x+w] #black & white version
            roi_color = frame[y:y+h, x:x+w]#Color version
             # ---------------------------------------
            # Eyes detection
            # ---------------------------------------
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
            #Then draw rectangles:
            #Each detected eye is represented as:(ex, ey, ew, eh)
            for (ex, ey, ew, eh) in eyes:#Loops through all detected eyes  
                cv2.rectangle(roi_color, (ex, ey),
                              #roi_color because we want to draw inside the face not the whole frame 
                              (ex+ew, ey+eh), (255, 0, 0), 2)#Draws blue rectangles around eyes
                cv2.putText(roi_color, "Eye", (ex, ey - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
             # ---------------------------------------
            # Nose detection (only if nose XML loaded)
            # ---------------------------------------
            #if not nose_cascade.empty():
              #  nose = nose_cascade.detectMultiScale(roi_gray, 1.2, 5)
                #for (nx, ny, nw, nh) in nose:
                    #cv2.rectangle(roi_color, (nx, ny),
                                # (nx+nw, ny+nh), (0,255,255), 2)
                    #cv2.putText(roi_color, "Nose", (nx, ny - 10),
                           #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                   # break
                  # ---------------------------------------
            # Mouth detection (using smile cascade)
            # ---------------------------------------
            mouth = smile_cascade.detectMultiScale(roi_gray, 1.4, 10)
            for (mx, my, mw, mh) in mouth:
                cv2.rectangle(roi_color, (mx, my),
                              (mx+mw, my+mh), (0,165,255), 2)
                cv2.putText(roi_color, "Mouth", (mx, my - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
                break  # detect only the first mouth
        cv2.imshow("Face + Features Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    # Streamlit App
def app():
    st.title("Face + Eyes + Nose + Mouth Detection App")
    st.write("Click the button to start detection")
    if st.button("Start"):
        detect_faces()

if __name__ == "__main__":
    app()



    