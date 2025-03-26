# PoC-Openvino-Gesture-Control

This project demonstrates the use of **hand gesture control** leveraging **MediaPipe** for hand tracking and **PyAutoGUI** for simulating keyboard and mouse interactions. The objective is to allow users to control a computer interface using hand gestures captured by a webcam.

## Features
- **Open Hand**: Pause or play media (Spacebar).
- **Closed Fist**: Close the active application (Alt+F4).
- **Pointing Gesture**: Control the mouse pointer using the index finger.
- **Thumbs Up**: Fast forward 10 seconds in media players (press 'L').
  
## Requirements

- Python 3.7-3.12
- `opencv-python==4.5.5.62`
- `mediapipe==0.8.10`
- `pyautogui==0.9.53`
- `numpy==1.21.2`
  
You can install the dependencies by running:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/PoC-Openvino-Gesture-Control.git
    cd PoC-Openvino-Gesture-Control
    ```

2. Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```
3. Run the script:

    ```bash
    python gesture_control.py
    ```

This will start the webcam and begin detecting hand gestures. 

## How it Works

- **Hand Tracking**: The script uses **MediaPipe** for real-time hand tracking via the webcam. The hand landmarks are captured and analyzed to recognize specific gestures such as `Pointing`, `Thumbs Up`, `Closed Fist`, and `Open Hand`.
  
- **Gesture Recognition**: Hand gestures are recognized using custom logic to identify whether the thumb is up, the fingers are curled, or the hand is open.
  
- **Actions**: Once a gesture is recognized, **PyAutoGUI** simulates actions such as pressing the spacebar, controlling the mouse, or closing active applications.

## Example Gestures
- **Open Hand**: Pauses or plays media.
- **Closed Fist**: Closes the active application window.
- **Pointing**: Controls the mouse cursor using the index finger.
- **Thumbs Up**: Fast forwards 10 seconds in supported media players.
