
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import pyautogui
import time
import csv
from collections import deque

# Optimize PyAutoGUI for real-time responsiveness
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False

def main():
    # Initialize hardware variables outside the try block for safe memory scoping
    cap = None
    
    try:
        # 1. Initialize ONNX Inference Engine for Gesture CNN
        model_path = "gesture_cnn.onnx"
        ort_session = ort.InferenceSession(model_path)
        input_name = ort_session.get_inputs()[0].name
        
        # Standard MediaPipe initialization
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.70
        )
        
        # 3. Configure Video Capture (640x480 @ 60 FPS Parity)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        if not cap.isOpened():
            print("CRITICAL ERROR: Cannot open webcam!")
            return

        print("Python Baseline System Online. Starting Telemetry Data Collection (10000 frames)...")
        
        # 4. Initialize Stability Buffers & Timers
        gesture_history = deque(maxlen=5) # HISTORY_SIZE = 5
        prev_x, prev_y = -1.0, -1.0
        last_action_time = time.time()
        lost_frames_count = 0
        last_id = -1
        
        # --- TELEMETRY INITIALIZATION ---
        inference_times = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Mirror frame processing
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Draw the 85% re-centered bounding box boundary
            box_size = int(min(w, h) * 0.85)
            offset_x = (w - box_size) // 2
            offset_y = (h - box_size) // 2
            cv2.rectangle(frame, (offset_x, offset_y), (offset_x + box_size, offset_y + box_size), (0, 255, 0), 2)
            
            # Convert color space for MediaPipe Processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
                lost_frames_count = 0 # Hand detected, reset anti-flicker buffer
                
                # Extract tracking coordinates
                hand_lms = results.multi_hand_landmarks[0]
                curr_x = hand_lms.landmark[9].x
                curr_y = hand_lms.landmark[9].y
                
                # Extract 3D World Landmarks for 1D CNN Inference
                world_lms = results.multi_hand_world_landmarks[0]
                temp_x = [lm.x for lm in world_lms.landmark]
                temp_y = [lm.y for lm in world_lms.landmark]
                temp_z = [lm.z for lm in world_lms.landmark]
                
                # Shift coordinates relative to Wrist
                wrist_x, wrist_y, wrist_z = temp_x[0], temp_y[0], temp_z[0]
                
                landmarks_vector = []
                for i in range(21):
                    landmarks_vector.append(temp_x[i] - wrist_x)
                    landmarks_vector.append(temp_y[i] - wrist_y)
                    landmarks_vector.append(temp_z[i] - wrist_z)
                
                # 5. TELEPORTATION GUARD
                if prev_x != -1.0:
                    distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    if distance > 0.15: 
                        prev_x, prev_y = curr_x, curr_y
                        cv2.imshow("AYNIYH Python Baseline Tracker", frame)
                        if cv2.waitKey(1) == 27: break
                        continue
                prev_x, prev_y = curr_x, curr_y

                # 6. GESTURE INFERENCE VIA ONNX RUNTIME (WITH TELEMETRY)
                brain_data = np.array(landmarks_vector, dtype=np.float32).reshape(1, 21, 3)
                ort_inputs = {input_name: brain_data}
                
                # --- START TELEMETRY TIMER ---
                start_time = time.perf_counter()
                probabilities = ort_session.run(None, ort_inputs)[0][0]
                end_time = time.perf_counter()
                # --- END TELEMETRY TIMER ---
                
                # Record Latency
                inference_ms = (end_time - start_time) * 1000
                inference_times.append(inference_ms)
                frame_count += 1
                
                raw_gesture = int(np.argmax(probabilities))
                max_p = probabilities[raw_gesture]
                
                if max_p > 0.60:
                    if raw_gesture != last_id:
                        print(f"[{frame_count}/10000] GESTURE ID {raw_gesture} DETECTED ({int(max_p * 100)}%) - Latency: {inference_ms:.2f}ms")
                        last_id = raw_gesture
                    gesture_history.append(raw_gesture)
                else:
                    gesture_history.append(-1)

                # Export conditions
                if frame_count >= 10000:
                    with open('python_latency.csv', 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Latency_ms'])
                        for t in inference_times:
                            writer.writerow([t])
                    print("Python Benchmark Complete. Data saved to 'python_latency.csv'")
                    break

                # 7. GESTURE VOTING DEBOUNCER
                votes = {}
                for g in gesture_history: votes[g] = votes.get(g, 0) + 1
                
                gesture = -1
                max_votes = 0
                for g, count in votes.items():
                    if count > max_votes:
                        max_votes = count
                        gesture = g
                        
                if max_votes < 3: gesture = -1

                # 8. OS MANIPULATION
                screen_w, screen_h = pyautogui.size()
                target_x = int(curr_x * screen_w)
                target_y = int(curr_y * screen_h)

                if gesture not in [0, 1]: pyautogui.moveTo(target_x, target_y)

                now = time.time()
                ms_since_last_action = (now - last_action_time) * 1000

                if gesture in [2, 3, 6, 7]: pyautogui.mouseUp(button='left')
                elif gesture == 0:
                    if ms_since_last_action > 400: pyautogui.click(button='left'); last_action_time = now
                elif gesture == 1:
                    if ms_since_last_action > 500: pyautogui.click(button='right'); last_action_time = now
                elif gesture == 10: pyautogui.mouseDown(button='left')
                elif gesture == 4:
                    if ms_since_last_action > 100: pyautogui.scroll(120); last_action_time = now
                elif gesture == 5:
                    if ms_since_last_action > 100: pyautogui.scroll(-120); last_action_time = now
                elif gesture == 8:
                    if ms_since_last_action > 150: pyautogui.press('volumeup'); last_action_time = now
                elif gesture == 9:
                    if ms_since_last_action > 150: pyautogui.press('volumedown'); last_action_time = now
                elif gesture == 11:
                    if ms_since_last_action > 1000: pyautogui.press('volumemute'); last_action_time = now
                else: pyautogui.mouseUp(button='left')
            else:
                lost_frames_count += 1
                if lost_frames_count > 5:
                    pyautogui.mouseUp(button='left')
                    prev_x, prev_y = -1.0, -1.0
                    gesture_history.clear()

            cv2.imshow("AYNIYH Python Baseline Tracker", frame)
            if cv2.waitKey(1) == 27: break
                
    except Exception as e:
        print(f"FATAL ERROR: {e}")
    finally:
        if cap is not None: cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
