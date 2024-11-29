import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam input
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get the screen size for mapping
screen_width, screen_height = pyautogui.size()

# Smoothing
smoothing_factor = 5
x_queue, y_queue = deque(maxlen=smoothing_factor), deque(maxlen=smoothing_factor)
 
# Gesture thresholds
click_threshold = 30
scroll_threshold = 40

# Variables to keep track of gestures
dragging = False
right_click_held = False
scroll_mode = False
last_scroll_time = 0
scroll_cooldown = 0.1  # seconds

# Helper functions
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def smooth_coordinates(x, y):
    x_queue.append(x)
    y_queue.append(y)
    return int(sum(x_queue) / len(x_queue)), int(sum(y_queue) / len(y_queue))

def draw_gesture_feedback(img, gesture, position):
    cv2.putText(img, gesture, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Main loop
while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to read frame from webcam.")
        break

    img = cv2.flip(img, 1)
    img_height, img_width, _ = img.shape

    # Convert the BGR image to RGB and process it with MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            landmarks = [(int(lm.x * img_width), int(lm.y * img_height)) for lm in hand_landmarks.landmark]

            # Extract relevant landmarks
            index_finger_tip = landmarks[8]
            thumb_tip = landmarks[4]
            middle_finger_tip = landmarks[12]
            ring_finger_tip = landmarks[16]

            # Map and smooth the index finger's position to screen coordinates
            screen_x = np.interp(index_finger_tip[0], (0, img_width), (0, screen_width))
            screen_y = np.interp(index_finger_tip[1], (0, img_height), (0, screen_height))
            smooth_x, smooth_y = smooth_coordinates(screen_x, screen_y)

            # Move the mouse cursor
            pyautogui.moveTo(smooth_x, smooth_y)

            # Calculate distances for gestures
            thumb_index_distance = calculate_distance(thumb_tip, index_finger_tip)
            thumb_middle_distance = calculate_distance(thumb_tip, middle_finger_tip)
            index_middle_distance = calculate_distance(index_finger_tip, middle_finger_tip)

            # Left click (pinch between thumb and index finger)
            if thumb_index_distance < click_threshold:
                if not dragging:
                    pyautogui.mouseDown(button='left')
                    dragging = True
                draw_gesture_feedback(img, "Left Click", (50, 50))
            else:
                if dragging:
                    pyautogui.mouseUp(button='left')
                    dragging = False

            # Right click (pinch between thumb and middle finger)
            if thumb_middle_distance < click_threshold:
                if not right_click_held:
                    pyautogui.click(button='right')
                    right_click_held = True
                draw_gesture_feedback(img, "Right Click", (50, 100))
            else:
                right_click_held = False

            # Scroll mode (index and middle fingers together, others folded)
            if index_middle_distance < scroll_threshold and \
               calculate_distance(ring_finger_tip, thumb_tip) < scroll_threshold:
                scroll_mode = True
                draw_gesture_feedback(img, "Scroll Mode", (50, 150))
            else:
                scroll_mode = False

            # Perform scrolling
            if scroll_mode:
                current_time = time.time()
                if current_time - last_scroll_time > scroll_cooldown:
                    if index_finger_tip[1] < img_height // 2:
                        pyautogui.scroll(1)  # Scroll up
                    else:
                        pyautogui.scroll(-1)  # Scroll down
                    last_scroll_time = current_time

    # Show the image
    cv2.imshow("Advanced Virtual Mouse", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources 
cap.release()
cv2.destroyAllWindows()