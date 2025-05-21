# app.py - Integrated Face Recognition and Emotion Detection
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import os
import time
import threading
import base64

# Import face_recognition with an alias to avoid conflicts
import face_recognition as fr

# Import TensorFlow for emotion detection
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Global variables
camera = None
output_frame = None
lock = threading.Lock()
processing_active = False

# Load emotion model
try:
    emotion_model = load_model('emotion_model.h5')
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    print("Emotion model loaded successfully")
except Exception as e:
    print(f"Error loading emotion model: {e}")
    emotion_model = None

# Face recognition variables
known_face_encodings = []
known_face_names = []

def load_known_faces(dataset_path="dataset"):
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Created dataset directory: {dataset_path}")
        return

    files_processed = 0
    faces_found = 0
    
    for root, dirs, files in os.walk(dataset_path):
        print(f"Found {len(files)} files in dataset")
        for file in files:
            if file.endswith(("jpg", "jpeg", "png")):
                try:
                    files_processed += 1
                    name = file.split("_")[0]
                    image_path = os.path.join(root, file)
                    print(f"Loading image: {image_path}")
                    
                    # Use OpenCV instead of face_recognition.load_image_file
                    image = cv2.imread(image_path)
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Get face encodings
                    face_locations = fr.face_locations(rgb_image)
                    if face_locations:
                        faces_found += 1
                        encoding = fr.face_encodings(rgb_image, face_locations)[0]
                        known_face_encodings.append(encoding)
                        known_face_names.append(name)
                    else:
                        print(f"No face found in image: {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    print(f"Processed {files_processed} files, found {faces_found} faces")
    print(f"Known faces: {set(known_face_names)}")

def process_frames():
    global output_frame, lock, processing_active
    
    # Load known faces
    load_known_faces()
    
    if not known_face_encodings:
        print("No known faces loaded. Detection will only find unknown faces.")
    
    # Initialize video capture
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Could not open webcam")
        processing_active = False
        return
    
    while processing_active:
        success, frame = camera.read()
        if not success:
            print("Failed to grab frame")
            break
            
        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = fr.face_locations(rgb_frame)
        encodings = fr.face_encodings(rgb_frame, faces)
        
        for (top, right, bottom, left), encoding in zip(faces, encodings):
            matches = []
            name = "Unknown"
            confidence = 0
            
            if known_face_encodings:
                # Use a slightly higher tolerance for better matching
                matches = fr.compare_faces(known_face_encodings, encoding, tolerance=0.6)
                face_distances = fr.face_distance(known_face_encodings, encoding)
                
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
                    cv2.putText(frame, "Unknown", (left, top-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No registered faces", (left, top-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Emotion detection
            if emotion_model is not None:
                try:
                    face_image = frame[top:bottom, left:right]
                    face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    face_image_resized = cv2.resize(face_image_gray, (48, 48))  
                    face_image_normalized = face_image_resized / 255.0
                    face_image_reshaped = np.reshape(face_image_normalized, (1, 48, 48, 1))  

                    preds = emotion_model.predict(face_image_reshaped)
                    emotion_idx = np.argmax(preds)
                    emotion = emotion_labels[emotion_idx]
                    emotion_confidence = preds[0][emotion_idx]
                    cv2.putText(frame, f"Emotion: {emotion} ({emotion_confidence*100:.1f}%)", 
                                (left, bottom+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error in emotion detection: {e}")
        
        # Update the output frame
        with lock:
            output_frame = frame.copy()
    
    camera.release()

def generate_frames():
    global output_frame, lock
    
    while True:
        with lock:
            if output_frame is None:
                continue
            
            # Encode the frame in JPEG format
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            
            if not flag:
                continue
        
        # Yield the output frame in byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')
        
        # Sleep to reduce CPU usage
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_processing', methods=['POST'])
def start_processing():
    global processing_active
    
    if not processing_active:
        processing_active = True
        threading.Thread(target=process_frames).start()
        return jsonify({"status": "success", "message": "Processing started"})
    
    return jsonify({"status": "info", "message": "Processing already active"})

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    global processing_active
    
    if processing_active:
        processing_active = False
        return jsonify({"status": "success", "message": "Processing stopped"})
    
    return jsonify({"status": "info", "message": "Processing already stopped"})

@app.route('/register_face', methods=['POST'])
def register_face():
    if 'name' not in request.form:
        return jsonify({"status": "error", "message": "Name is required"})
    
    name = request.form['name']
    image_data = request.form.get('image')
    
    if not image_data:
        return jsonify({"status": "error", "message": "Image data is required"})
    
    try:
        print(f"Received image data length: {len(image_data) if image_data else 'None'}")
        
        # Fix: Make sure we're properly extracting the base64 data
        if "base64," in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        print(f"Decoded image bytes length: {len(image_bytes)}")
        
        np_arr = np.frombuffer(image_bytes, np.uint8)
        print(f"NumPy array shape: {np_arr.shape if np_arr.size > 0 else 'Empty array'}")
        
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Check if decoding was successful
        if image is None:
            print("Image decoding failed")
            return jsonify({"status": "error", "message": "Failed to decode image data"})
        
        print(f"Decoded image shape: {image.shape}")
        
        # Check if there's a face in the image before saving
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = fr.face_locations(rgb_image)
        
        if not face_locations:
            return jsonify({
                "status": "error", 
                "message": "No face detected in the image. Please try again with a clearer face image."
            })
        
        # Extract and save just the face region for better recognition
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]
        
        # Create dataset directory if it doesn't exist
        if not os.path.exists("dataset"):
            os.makedirs("dataset")
        
        # Save the face image
        file_path = os.path.join("dataset", f"{name}_{int(time.time())}.jpg")
        success = cv2.imwrite(file_path, face_image)
        
        if not success:
            return jsonify({"status": "error", "message": "Failed to save image file"})
        
        # Verify the file was saved
        if not os.path.exists(file_path):
            return jsonify({"status": "error", "message": "File was not saved properly"})
            
        print(f"Face image saved to {file_path}")
        
        # Reload known faces
        load_known_faces()
        
        return jsonify({"status": "success", "message": "Face registered successfully"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Error registering face: {str(e)}"})

@app.route('/get_registered_faces')
def get_registered_faces():
    load_known_faces()
    unique_names = list(set(known_face_names))
    return jsonify({
        "status": "success", 
        "count": len(known_face_names),
        "unique_count": len(unique_names),
        "names": unique_names
    })

if __name__ == '__main__':
    # Load faces on startup
    load_known_faces()
    app.run(debug=True)
