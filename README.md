# Basic_face_liveness_detection

## Overview
This is a simple web application that performs **face liveness detection** by analyzing a video feed to verify whether the face shown is live or fake. It performs multiple checks such as brightness, background consistency, detecting a single face, blink detection, smile detection, and nod detection.

The frontend displays the live video feed along with prompts and stage-by-stage status updates, finally showing a live/fake verdict with confidence score.

---

## Features
- Real-time face liveness detection with multiple stages  
- Clear user prompts to guide the user through the detection steps  
- Stage-wise status updates with pass/fail/pending states  
- Final verdict display with confidence score  
- Simple, clean UI with responsive status updates

---

## Installation

1. (Optional) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

2. pip install -r requirements.txt

3. Download shape_predictor_68_face_landmarks.dat
   abstract and add in your project

4. Run the project

python app.py

5. File Structure

├── app.py               
├── templates/
│   └── index.html      
├── static/
│   └── styles.css    
├── requirements.txt     
└── README.md           
