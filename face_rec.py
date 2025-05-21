# Importing Dependencies
import face_recognition
import cv2
import numpy as np
import os
import json
import tkinter as tk
from tkinter import simpledialog, messagebox, Label
from tkinter import Button, ttk
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

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        return

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
        messagebox.showerror("Error", "Could not open webcam")
        return

    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_frame)
        
        # Display progress
        cv2.putText(frame, f"Capturing: {count}/{num_samples}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if faces:
            top, right, bottom, left = faces[0]
            face_image = rgb_frame[top:bottom, left:right]
            face_image_resized = cv2.resize(face_image, (200, 200)) 
            file_path = os.path.join(save_dir, f"{name}_{count}.jpg")
            cv2.imwrite(file_path, face_image_resized)
            count += 1
            
            # Draw rectangle around detected face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
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
    
    if not known_face_encodings:
        messagebox.showwarning("Warning", "No faces registered yet. Please register a face first.")
        return
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam")
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
            
            # Calculate face distance for better confidence measure
            face_distances = face_recognition.face_distance(known_face_encodings, encoding)
            
            if True in matches:
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]
                confidence_percentage = f"{confidence*100:.1f}%"
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    cv2.putText(frame, f"{name} ({confidence_percentage})", (left, top-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown - Register Needed", (left, top-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Emotion detection
            try:
                face_image = frame[top:bottom, left:right]
                face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image_resized = cv2.resize(face_image_gray, (48, 48))  
                face_image_resized = face_image_resized.astype("float") / 255.0
                face_image_resized = np.expand_dims(face_image_resized, axis=-1)  
                face_image_resized = np.expand_dims(face_image_resized, axis=0)  

                preds = emotion_model.predict(face_image_resized)
                emotion_idx = np.argmax(preds)
                emotion = emotion_labels[emotion_idx]
                emotion_confidence = preds[0][emotion_idx]
                cv2.putText(frame, f"Emotion: {emotion} ({emotion_confidence*100:.1f}%)", 
                            (left, bottom+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error in emotion detection: {e}")

        cv2.imshow('Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Improved UI
def main():
    root = tk.Tk()
    root.title("Face Recognition System")
    root.geometry("500x400")
    root.configure(bg="#f0f0f0")
    
    # Load known faces on startup
    load_known_faces()
    
    title_label = Label(root, text="Face Recognition & Emotion Detection", 
                        font=("Arial", 16, "bold"), bg="#f0f0f0")
    title_label.pack(pady=20)
 
    instructions = Label(root, text="Register your face or start recognition", 
                         font=("Arial", 12), bg="#f0f0f0")
    instructions.pack(pady=10)

    # Frame for buttons
    button_frame = tk.Frame(root, bg="#f0f0f0")
    button_frame.pack(pady=20)

    register_button = ttk.Button(button_frame, text="Register Face", command=register_face)
    register_button.grid(row=0, column=0, padx=10, pady=10)

    recognize_button = ttk.Button(button_frame, text="Recognize Face", command=recognize_faces)
    recognize_button.grid(row=0, column=1, padx=10, pady=10)

    # Status
    status_label = Label(root, text=f"Registered faces: {len(known_face_names)}", 
                         font=("Arial", 10), bg="#f0f0f0")
    status_label.pack(pady=10)
    
    # Instructions
    help_text = """
    Instructions:
    1. Click 'Register Face' to add yourself to the database
    2. Click 'Recognize Face' to start face and emotion detection
    3. Press 'q' to exit the camera view
    """
    help_label = Label(root, text=help_text, justify=tk.LEFT, 
                       font=("Arial", 10), bg="#f0f0f0")
    help_label.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
