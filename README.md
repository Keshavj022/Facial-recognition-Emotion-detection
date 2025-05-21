# Facial Recognition and Emotion Detection System

## Project Description
This project implements a robust facial recognition and emotion detection system using advanced computer vision and deep learning techniques. The system captures facial images, recognizes faces from a pre-registered dataset, and detects the emotional state of individuals in real-time. Built with Python, it leverages libraries such as OpenCV, face_recognition, and TensorFlow/Keras to achieve high accuracy and efficiency. The system features both a desktop GUI application and a modern web interface for versatile deployment options.

## Key Features
- **Real-Time Face Detection:** Detects and recognizes faces using advanced face recognition algorithms with confidence scoring.
- **Emotion Detection:** Identifies seven emotional states (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) using a CNN-based deep learning model.
- **Dual Interface Options:** 
  - Desktop GUI application built with Tkinter
  - Modern web interface built with Flask and Bootstrap
- **User Registration System:** Simple process to add new users to the recognition database.
- **Responsive Design:** Web interface adapts to various screen sizes for optimal user experience.

## Technologies Used
- **Python:** Primary programming language for backend development.
- **OpenCV:** Computer vision library for image processing and face detection.
- **face_recognition:** High-level face recognition library based on dlib.
- **TensorFlow/Keras:** For building and running the emotion detection CNN model.
- **Flask:** Web framework for the browser-based interface.
- **Bootstrap:** Frontend framework for responsive and attractive UI design.
- **JavaScript:** For interactive elements in the web interface.

## Installation
### Prerequisites
- Python 3.8+
- Webcam access
- Required Python libraries (see `requirements.txt`)

### Installation Steps
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/Keshavj022/Facial-recognition-Emotion-detection.git
    cd Facial-recognition-Emotion-detection
    ```
2. **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Download Pre-trained Models:**
    - Ensure the emotion detection model (`emotion_model.h5`) is in the project directory.

## Usage
### Desktop Application
1. Run the desktop application:
    ```bash
    python face_recognition_app.py
    ```
2. Use the "Register Face" button to add new faces to the system.
3. Click "Recognize Face" to start real-time recognition and emotion detection.

### Web Interface
1. Start the web server:
    ```bash
    python app.py
    ```
2. Open your browser and navigate to `http://127.0.0.1:5000`
3. Click "Register Face" to capture and register a new face.
4. Use "Start Detection" to begin real-time face recognition and emotion analysis.

## System Architecture
- **Face Registration Module:** Captures and processes facial images for the recognition database.
- **Recognition Engine:** Identifies registered faces and calculates confidence scores.
- **Emotion Analysis Module:** Processes facial expressions to determine emotional states.
- **User Interface Layer:** Provides interaction through either desktop GUI or web interface.

## Code Overview
- `face_recognition_app.py`: Desktop application with Tkinter GUI.
- `app.py`: Flask web application server.
- `templates/index.html`: Web interface template.
- `emotion_model.py`: Script for training the emotion detection model.

## Performance Considerations
- For optimal performance, ensure good lighting conditions.
- System performance depends on hardware capabilities, particularly for real-time processing.
- The web interface requires a modern browser with webcam access permissions.

## Future Enhancements
- Multi-face tracking and recognition
- Emotion trend analysis over time
- Integration with authentication systems
- Mobile application support

## Contributing
Contributions are welcome! Please feel free to submit issues or pull requests. For detailed instructions, refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.