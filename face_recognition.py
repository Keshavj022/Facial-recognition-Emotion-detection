# Importing Dependencies
import face_recognition
import cv2
import numpy as np
import os
import json
import tkinter as tk
from tkinter import simpledialog, messagebox, Label
from tkinter import Button
from keras.models import load_model

# Loading pre-trained models
emotion_model = load_model('emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initializing face recognition variables
known_face_encodings = []
known_face_names = []

# Loading known faces
def load_known_faces(dataset_path="dataset"):
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith("jpg"):
                name = file.split("_")[0]
                image_path = os.path.join(root, file)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    encoding = encodings[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)
                else:
                    print(f"No face found in image: {file}")              

# Captures 50 photos and save in Grayscale
def capture_images(name, num_samples=50, save_dir="dataset"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_frame)
        
        for face in faces:
            top, right, bottom, left = face
            face_image = rgb_frame[top:bottom, left:right]
            face_image_resized = cv2.resize(face_image, (200, 200)) 
            file_path = os.path.join(save_dir, f"{name}_{count}.jpg")
            cv2.imwrite(file_path, face_image_resized)
            count += 1
        
        cv2.imshow('Capture Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# New users registration
def register_face():
    name = simpledialog.askstring("Input", "Enter your name:")
    if name:
        if os.path.exists("dataset"):
            existing_names = {file.split("_")[0] for file in os.listdir("dataset") if file.endswith("jpg")}
            if name in existing_names:
                messagebox.showwarning("Warning", "Name already exists!")
                return
        
        capture_images(name)
        load_known_faces()
        messagebox.showinfo("Success", "Face registered successfully!")

# Recognizing face
def recognize_faces():
    load_known_faces()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, faces)

        for (top, right, bottom, left), encoding in zip(faces, encodings):
            matches = face_recognition.compare_faces(known_face_encodings, encoding)
            name = "Unknown"
            confidence = 0

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                confidence = 1.0  # Confidence is maximum if a match is found
                cv2.putText(frame, f"Hello {name}", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown - Register Needed", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            face_image = frame[top:bottom, left:right]
            face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image_resized = cv2.resize(face_image_gray, (48, 48))  
            face_image_resized = face_image_resized.astype("float") / 255.0
            face_image_resized = np.expand_dims(face_image_resized, axis=-1)  
            face_image_resized = np.expand_dims(face_image_resized, axis=0)  

            preds = emotion_model.predict(face_image_resized)
            emotion = emotion_labels[np.argmax(preds)]
            cv2.putText(frame, f"Emotion: {emotion}", (left, bottom+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    global detector
    root = tk.Tk()
    root.title("Face Recognition System")
 
    instructions = Label(root, text="Press 'Register Face' to add a new face. Press 'Recognize Face'.")
    instructions.pack(pady=10)

    register_button = Button(root, text="Register Face", command=register_face)
    register_button.pack(pady=20)

    recognize_button = Button(root, text="Recognize Face", command=recognize_faces)
    recognize_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
