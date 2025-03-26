import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe Hands with higher confidence thresholds
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.7,  # Confidence threshold for detection hands
    min_tracking_confidence=0.7,   # Confidence threshold for tracking hands
    model_complexity=1             
)
mp_draw = mp.solutions.drawing_utils

# Set up PyAutoGUI
pyautogui.FAILSAFE = True  # Move cursor to upper-left corner to abort
screen_width, screen_height = pyautogui.size()

# Variables for gesture control
prev_gesture = None
gesture_start_time = 0
gesture_duration = 0.5  # Hold gesture for this many seconds to trigger action
pointing_active = False
last_action_time = 0
action_cooldown = 1.0  # Time between actions in seconds

# Gesture history for smoothing
gesture_history = []
history_length = 5

# Function to recognize gestures with improved accuracy
def recognize_gesture(landmarks):
    # Define key landmark points
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]
    
    # Calculate vertical distances (accounting for hand orientation)
    # Using distance from MCP to tip normalized by distance from wrist to MCP for scale invariance
    
    # Thumb is up if the tip is sufficiently above the MCP
    thumb_mcp_to_tip_y = thumb_mcp.y - thumb_tip.y
    wrist_to_thumb_mcp_y = wrist.y - thumb_mcp.y
    thumb_up_ratio = thumb_mcp_to_tip_y / max(wrist_to_thumb_mcp_y, 0.001)
    thumb_up = thumb_up_ratio > 0.4
    
    # For other fingers, check if tip is higher than pip and calculate ratio
    index_pip_to_tip_y = index_pip.y - index_tip.y
    index_mcp_to_pip_y = index_mcp.y - index_pip.y
    index_up_ratio = index_pip_to_tip_y / max(index_mcp_to_pip_y, 0.001)
    index_up = index_up_ratio > 0.6
    
    middle_pip_to_tip_y = middle_pip.y - middle_tip.y
    middle_mcp_to_pip_y = middle_mcp.y - middle_pip.y
    middle_up_ratio = middle_pip_to_tip_y / max(middle_mcp_to_pip_y, 0.001)
    middle_up = middle_up_ratio > 0.6
    
    ring_pip_to_tip_y = ring_pip.y - ring_tip.y
    ring_mcp_to_pip_y = ring_mcp.y - ring_pip.y
    ring_up_ratio = ring_pip_to_tip_y / max(ring_mcp_to_pip_y, 0.001)
    ring_up = ring_up_ratio > 0.6
    
    pinky_pip_to_tip_y = pinky_pip.y - pinky_tip.y
    pinky_mcp_to_pip_y = pinky_mcp.y - pinky_pip.y
    pinky_up_ratio = pinky_pip_to_tip_y / max(pinky_mcp_to_pip_y, 0.001)
    pinky_up = pinky_up_ratio > 0.6
    
    # Calculate horizontal distance between thumb and index (normalized by hand size)
    hand_width = max(abs(thumb_mcp.x - pinky_mcp.x), 0.001)
    thumb_index_distance = abs(thumb_tip.x - index_tip.x) / hand_width
    
    # Closed fist detection with improved logic
    # Check if all fingers are curled (not extended) and thumb is close to fingers
    is_closed_fist = (
        not index_up and 
        not middle_up and 
        not ring_up and 
        not pinky_up and
        thumb_index_distance < 0.3  # Thumb should be close to the index finger
    )
    
    # Pointing gesture - only index finger is extended
    is_pointing = (
        index_up and 
        not middle_up and 
        not ring_up and 
        not pinky_up
    )
    
    # Thumbs up gesture - only thumb is extended and pointing upward
    is_thumbs_up = (
        thumb_up and 
        not index_up and 
        not middle_up and 
        not ring_up and 
        not pinky_up
    )
    
    # Open hand - all fingers are extended
    is_open_hand = (
        index_up and 
        middle_up and 
        ring_up and 
        pinky_up
    )
    
    # Confidence scores for each gesture
    gestures = {
        "Closed Fist": 1.0 if is_closed_fist else 0.0,
        "Open Hand": 1.0 if is_open_hand else 0.0,
        "Pointing": 1.0 if is_pointing else 0.0,
        "Thumbs Up": 1.0 if is_thumbs_up else 0.0
    }
    
    # Get the most likely gesture
    max_gesture = max(gestures.items(), key=lambda x: x[1])
    
    if max_gesture[1] > 0.0:
        return max_gesture[0]
    else:
        return "Unknown Gesture"

# Function to smooth gesture recognition using history
def smooth_gesture(current_gesture):
    global gesture_history
    
    # Add current gesture to history
    gesture_history.append(current_gesture)
    
    # Keep history at fixed length
    if len(gesture_history) > history_length:
        gesture_history = gesture_history[-history_length:]
    
    # Count occurrences of each gesture in history
    gesture_counts = {}
    for g in gesture_history:
        if g in gesture_counts:
            gesture_counts[g] += 1
        else:
            gesture_counts[g] = 1
    
    # Return most common gesture if it appears more than 40% of the time
    most_common = max(gesture_counts.items(), key=lambda x: x[1])
    if most_common[1] / len(gesture_history) > 0.4:
        return most_common[0]
    else:
        return current_gesture  # Return current if no clear winner

# Function to perform actions based on gestures
def perform_action(gesture, landmarks=None):
    
    global pointing_active, last_action_time
    
    current_time = time.time()
    if current_time - last_action_time < action_cooldown:
        return
    
    if gesture == "Open Hand":
        print("Action: Pause media")
        pyautogui.press('space')  # Space bar to pause/play most media players
        last_action_time = current_time
        
    elif gesture == "Closed Fist":
        print("Action: Close active application")
        pyautogui.hotkey('alt', 'f4')  # Alt+F4 to close active window
        last_action_time = current_time
        
    elif gesture == "Pointing":
        if not pointing_active:
            pointing_active = True
            print("Action: Mouse control activated")
            
        # Use index finger for mouse control with improved smoothing
        if landmarks:
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = landmarks[mp_hands.HandLandmark.WRIST]
            
            # Calculate hand size for scaling
            hand_width = abs(landmarks[mp_hands.HandLandmark.THUMB_MCP].x - 
                            landmarks[mp_hands.HandLandmark.PINKY_MCP].x)
            
            # Map hand coordinates to screen coordinates with dynamic scaling
            # Use relative position from wrist to index tip for more intuitive control
            mouse_x = int(screen_width * (1.0 - index_tip.x))  # Flip X for mirror effect
            mouse_y = int(screen_height * index_tip.y * 0.7)  # Scale Y for comfort
            
            # Apply smoothing to mouse movement
            # Use exponential smoothing for mouse position
            current_mouse_x, current_mouse_y = pyautogui.position()
            smoothing_factor = 0.3  # Lower = smoother but more lag
            
            smooth_x = int(current_mouse_x * (1 - smoothing_factor) + mouse_x * smoothing_factor)
            smooth_y = int(current_mouse_y * (1 - smoothing_factor) + mouse_y * smoothing_factor)
            
            # Ensure coordinates are within screen bounds
            smooth_x = max(0, min(smooth_x, screen_width))
            smooth_y = max(0, min(smooth_y, screen_height))
            
            # Move mouse pointer
            pyautogui.moveTo(smooth_x, smooth_y, duration=0.1)
            
    elif gesture == "Thumbs Up":
        print("Action: Fast Forward 10 secs")
        pyautogui.press('l')  # press l to fast forward 10 secs
        last_action_time = current_time
    
    else:
        pointing_active = False

# Initialize webcam
cap = cv2.VideoCapture(0)

# Try to set higher resolution for better tracking
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a more intuitive mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    result = hands.process(rgb_frame)
    
    current_gesture = "No Hand Detected"
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Recognize gesture
            raw_gesture = recognize_gesture(hand_landmarks.landmark)
            
            # Apply smoothing to gesture recognition
            current_gesture = smooth_gesture(raw_gesture)
            
            # Display the gesture
            cv2.putText(frame, current_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Check if gesture is stable
            if current_gesture == prev_gesture:
                if time.time() - gesture_start_time > gesture_duration:
                    perform_action(current_gesture, hand_landmarks.landmark)
            else:
                # Reset the timer for the new gesture
                gesture_start_time = time.time()
            
            prev_gesture = current_gesture
    else:
        cv2.putText(frame, "No Hand Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        pointing_active = False
        gesture_history = []  # Clear gesture history when no hand is detected
    
    # Add control instructions to the frame
    instructions = [
        "Open Hand = Pause/Play",
        "Closed Fist = Close App",
        "Pointing = Mouse Control",
        "Thumbs Up = Fast Forward 10 secs"
    ]
    
    y_offset = 70
    for instruction in instructions:
        cv2.putText(frame, instruction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        y_offset += 40
    
    # Resize and Display the frame
    display_frame = cv2.resize(frame, (720, 500)) 
    cv2.imshow("Hand Gesture Control", display_frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

