# Facial Recognition and Emotion Detection System

## Project Description
This project implements a robust facial recognition and emotion detection system using advanced computer vision and deep learning techniques. The system is designed to capture facial images, recognize faces from a pre-registered dataset, and detect the emotional state of the person in real-time. It is built with Python and utilizes libraries such as OpenCV, dlib, face_recognition, and Keras for achieving high accuracy and efficiency.

## Key Features
- **Real-Time Face Detection:** Detects and recognizes faces using advanced face recognition algorithms.
- **Emotion Detection:** Identifies emotional states (e.g., Angry, Happy, Sad) using a pre-trained deep learning model.
- **User-Friendly GUI:** Provides an intuitive graphical user interface for face registration and recognition.
- **Efficient Performance:** Handles real-time processing with minimal latency.

## Technologies Used
- **Python:** Programming language used for development.
- **OpenCV:** Library for computer vision tasks.
- **dlib:** For facial landmark detection.
- **face_recognition:** Advanced face recognition library.
- **Keras:** For emotion detection using a deep learning model.
- **Tkinter:** For creating the graphical user interface (GUI).

## Installation
### Prerequisites
- Python 3.x
- Required Python libraries (see `requirements.txt`)

### Installation Steps
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/Keshavj022/Facial-recognition-Emotion-detection.git
    cd facial-recognition-emotion-detection
    ```
2. **Install Required Libraries:**
    You can use pip to install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. **Download Pre-trained Models:**
    - Emotion Detection Model
    - Dlib Shape Predictor
    Place these files in the project directory.

## Usage
### 1. Register a New Face
To register a new face, use the GUI application:
1. Run the application:
    ```bash
    python app.py
    ```
2. Click on "Register Face" and enter the name of the person.
3. The system will capture images and save them for training.

### 2. Recognize Faces
To recognize faces:
1. Run the application:
    ```bash
    python app.py
    ```
2. Click on "Recognize Face" to start the face recognition and emotion detection process.

## GUI Overview
- **Register Face Button:** Registers a new face by capturing images and saving them for training.
- **Recognize Face Button:** Starts real-time face and emotion recognition.

## Code Overview
- `face_recognition.py`: Contains functions for face encoding, recognition, and emotion detection.
- `app.py`: The main GUI application script using Tkinter.
- `requirements.txt`: Lists the required Python libraries.

## Contributing
Feel free to contribute to this project by submitting issues or pull requests. For detailed instructions on how to contribute, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- OpenCV
- dlib
- face_recognition
- Keras
- Tkinter
