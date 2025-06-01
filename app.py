from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import time
import mediapipe as mp
from collections import OrderedDict

app = Flask(__name__) #initialize the Flask app.

# Detection stage setup
#----List of stages to check in order, with the maximum time allowed (in seconds) for each stage.---
STAGE_SEQUENCE = ['brightness', 'background', 'single_face', 'blink', 'smile', 'nod']
STAGE_TIMEOUTS = {
    'brightness': 5,
    'background': 5,
    'single_face': 6,
    'blink': 6,
    'smile': 8,
    'nod': 8
}

# MediaPipe and OpenCV setup
mp_face_mesh = mp.solutions.face_mesh
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# State initializer (to track progress, time, results, and nod/smile detection flags.)
state = {}
def initialize_state():
    now = time.time()
    return {
        'stage_index': 0,  # which stage we are at
        'stage_start_time': now, # when current stage started
        'stage_results': OrderedDict((stage, {'status': 'pending', 'start_time': now, 'done': False}) for stage in STAGE_SEQUENCE),
        'final_verdict': 'WAIT', # overall result (WAIT / LIVE / FAKE)
        'confidence': 0,  # percentage of stages passed
        'pitch_prev': None, # previous pitch value for nod detection
        'nod_direction': None, # current direction of nod
        'nod_count': 0,  # how many nods detected
        'stationary_count': 0,  # how long head stayed still
        'smile_allowed': False, # whether smile detection is allowed now
    
    }
state = initialize_state()

#  Returns True if the average brightness of the grayscale image is above 80 (bright enough).
def brightness_check(gray):
    return np.mean(gray) >= 80 

#Checks if background above face is varied enough (not too plain) by calculating standard deviation of pixel values.
def background_check(frame, face_rect):
    x, y, w, h = face_rect
    bg = frame[max(0, y - h // 2):y, x:x + w]
    return np.std(bg) > 15 if bg.size > 0 else False

#Calculates head tilt angle by using the vector between nose and chin landmarks. Helps detect nodding.
def calculate_pitch(landmarks, img_shape):
    h, w = img_shape[:2]
    nose = landmarks[1]
    chin = landmarks[152]
    nose_pt = np.array([nose.x * w, nose.y * h, nose.z * w])
    chin_pt = np.array([chin.x * w, chin.y * h, chin.z * w])
    vec = nose_pt - chin_pt
    angle_rad = np.arccos(np.clip(np.dot(vec / np.linalg.norm(vec), [0, -1, 0]), -1, 1))
    return np.degrees(angle_rad) - 90


#---HEAD NOD DETECTION
#---Tracks head tilt changes to detect nodding (up and down movements). Returns True after detecting two nods.
def detect_nod(pitch):
    THRESHOLD_DOWN = 2.0    
    THRESHOLD_UP = -2.0
    STATIONARY_THRESHOLD = 0.4
    STATIONARY_FRAMES_REQUIRED = 4

    # Initialize state keys if missing
    for key in ['pitch_prev', 'nod_direction', 'nod_count', 'stationary_count']:
        if key not in state:
            state[key] = None if key != 'nod_count' and key != 'stationary_count' else 0

    if state['pitch_prev'] is None:
        state['pitch_prev'] = pitch
        return False

    diff = pitch - state['pitch_prev']

    # Count frames where pitch barely changes (head stationary)
    if abs(diff) < STATIONARY_THRESHOLD:
        state['stationary_count'] = (state['stationary_count'] or 0) + 1
    else:
        state['stationary_count'] = 0

    # Detect nod up/down movement
    if state['stationary_count'] < STATIONARY_FRAMES_REQUIRED:
        if state['nod_direction'] is None:
            if diff > THRESHOLD_DOWN:
                state['nod_direction'] = 'down'
            elif diff < THRESHOLD_UP:
                state['nod_direction'] = 'up'
        else:
            if state['nod_direction'] == 'down' and diff < THRESHOLD_UP:
                state['nod_count'] += 1
                state['nod_direction'] = 'up'
            elif state['nod_direction'] == 'up' and diff > THRESHOLD_DOWN:
                state['nod_count'] += 1
                state['nod_direction'] = 'down'
    else:
        # Consider reset nod direction/count if head is stationary too long
        state['nod_direction'] = None
        state['nod_count'] = 0

    state['pitch_prev'] = pitch

    # Return True if nod count >= 2 (two nods detected)
    return state['nod_count'] >= 2


#---BLINK DETECTION---
# Measures eye aspect ratio (height/width) for both eyes. If the ratio is very low (eyes closed), returns True (blink detected).
def detect_blink(landmarks):
    left = [landmarks[362], landmarks[385], landmarks[387], landmarks[263], landmarks[373], landmarks[380]]
    right = [landmarks[33], landmarks[160], landmarks[158], landmarks[133], landmarks[153], landmarks[144]]
    def eye_ratio(eye):
        ver = np.linalg.norm([eye[1].x - eye[5].x, eye[1].y - eye[5].y]) + \
              np.linalg.norm([eye[2].x - eye[4].x, eye[2].y - eye[4].y])
        hor = np.linalg.norm([eye[0].x - eye[3].x, eye[0].y - eye[3].y])
        return ver / (2.0 * hor)
    return eye_ratio(left) < 0.25 and eye_ratio(right) < 0.25


#---SMILE DETECTION---
# Compares mouth width and height to decide if smiling. Only checks if smile detection is currently allowed.
def detect_smile(landmarks):
    if not state['smile_allowed']:
        return False

    left = np.array([landmarks[61].x, landmarks[61].y])
    right = np.array([landmarks[291].x, landmarks[291].y])
    top = np.array([landmarks[13].x, landmarks[13].y])
    bottom = np.array([landmarks[14].x, landmarks[14].y])

    horizontal_dist = np.linalg.norm(right - left)
    vertical_opening = np.linalg.norm(top - bottom)

    # Stricter condition to reduce false smile detections
    is_smile = horizontal_dist > 1.5 * vertical_opening and vertical_opening > 0.02

    if is_smile:
        # Reset flag to avoid multiple detections
        state['smile_allowed'] = False # prevent multiple detections
        return True

    return False


# Calls the appropriate check function based on current stage and returns True or False.
def process_detection(stage_name, current_time, gray, frame, faces, landmarks, pitch):
    
    if stage_name == 'brightness':
        return brightness_check(gray)
    elif stage_name == 'background':
        # Only pass if exactly one face
        return len(faces) == 1 and background_check(frame, faces[0])
    elif stage_name == 'single_face':
        # Fail if no face or more than one face
        if len(faces) == 1:
            return True
        else:
            # Draw warning on frame if multiple faces
            if len(faces) > 1:
                cv2.putText(frame, f"Multiple faces detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            elif len(faces) == 0:
                cv2.putText(frame, f"No face detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            return False
    elif stage_name == 'blink' and landmarks:
        return detect_blink(landmarks)
    elif stage_name == 'smile' and landmarks:
        return detect_smile(landmarks)
    elif stage_name == 'nod' and pitch is not None:
        return detect_nod(pitch)
    return False



# UPDATE DETECTION PROGRESS
def update_detection_state(state, current_time, gray, frame, faces, landmarks, pitch):
    
    if state['final_verdict'] != 'WAIT':
        return

    stage = STAGE_SEQUENCE[state['stage_index']]
    stage_info = state['stage_results'][stage]
    elapsed = current_time - stage_info['start_time']

    passed = process_detection(stage, current_time, gray, frame, faces, landmarks, pitch)

    if passed:
        stage_info['status'] = 'pass'
        stage_info['done'] = True
        if stage == 'blink':
         state['smile_allowed'] = True
        state['stage_index'] += 1
        if state['stage_index'] < len(STAGE_SEQUENCE):
            next_stage = STAGE_SEQUENCE[state['stage_index']]
            state['stage_results'][next_stage]['start_time'] = current_time
    elif elapsed > STAGE_TIMEOUTS[stage]:
        stage_info['status'] = 'fail'
        stage_info['done'] = True
        state['final_verdict'] = 'FAKE'

    if all(info['done'] for info in state['stage_results'].values()):
        if state['final_verdict'] != 'FAKE':
            state['final_verdict'] = 'LIVE'
            score = sum(1 for s in state['stage_results'].values() if s['status'] == 'pass')
            state['confidence'] = int((score / len(STAGE_SEQUENCE)) * 100)


#Show stage results, overall result, and confidence on the video feed.
def draw_status(frame, state):
    y = 30
    for stage, result in state['stage_results'].items():
        color = (0, 255, 0) if result['status'] == 'pass' else (0, 0, 255) if result['status'] == 'fail' else (255, 255, 255)
        cv2.putText(frame, f"{stage}: {result['status']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 30

    verdict_color = (0, 255, 0) if state['final_verdict'] == 'LIVE' else (0, 0, 255)
    cv2.putText(frame, f"Result: {state['final_verdict']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, verdict_color, 3)
    y += 40
    cv2.putText(frame, f"Confidence: {state['confidence']}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


#---Main video loop---
def gen_frames():
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(refine_landmarks=True) as mesh:
        while True:
            success, frame = cap.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mesh.process(frame_rgb)
            landmarks = results.multi_face_landmarks[0].landmark if results.multi_face_landmarks else None
            pitch = calculate_pitch(landmarks, frame.shape) if landmarks else None

            now = time.time()
            update_detection_state(state, now, gray, frame, faces, landmarks, pitch)

            y_offset = 30
            for stage, result in state['stage_results'].items():
                status = result['status']
                color = (0, 255, 0) if status == 'pass' else ((0, 0, 255) if status == 'fail' else (200, 200, 200))
                cv2.putText(frame, f"{stage}: {status.upper()}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25

            cv2.putText(frame, f"Result: {state['final_verdict']}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0,128,0) if state['final_verdict']=='LIVE' else (0,0,255), 3)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

#Serves the webpage with video feed.
@app.route('/')
def index():
    return render_template('index.html')

#Streams video frames continuously to the webpage.
@app.route('/video_feed')
def video_feed():
    global state
    state = initialize_state()  # reset state on each load
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#Returns current detection status and result as JSON, useful for JavaScript or external apps.
@app.route('/status')
def status():
    return jsonify({
        'stage_results': {k: v['status'] for k, v in state['stage_results'].items()},
        'confidence': state['confidence'],
        'final_verdict': state['final_verdict'],
    })

#Run the Flask app in debug mode when we execute this script.
if __name__ == '__main__':
    app.run(debug=True)