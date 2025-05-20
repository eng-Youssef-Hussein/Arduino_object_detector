import cv2

from ultralytics import YOLO

import serial  # For communication with Arduino

import time

import pyttsx3
 
def speak(text):

    engine = pyttsx3.init()

    engine.say(text)

    engine.runAndWait()
 
# === Initialize serial communication with Arduino ===

try:

    arduino = serial.Serial('COM1', 115200, timeout=1)  # ← CHANGE COM3 if needed

    time.sleep(2)  # Wait for connection

    print("✅ Arduino connected.")

except serial.SerialException as e:

    arduino = None

    print(f"❌ Arduino not connected: {e}. Proceeding without serial communication.")
 
# Load the YOLOv8 face detection model

model = YOLO("YOLO-weights/yolov8l-face-lindevs.pt")
 
# Open IP camera (or use 0 for laptop camera)

cap = cv2.VideoCapture('http://192.168.1.3:81/stream')  # ESP32-CAM stream
 
# Initialize counters

count = 0

previous_count = 0

failure_count = 0

max_retries = 10  # Number of allowed failures before exit
 
while cap.isOpened():

    ret, frame = cap.read()
 
    if not ret or frame is None or frame.size == 0:

        print(f"⚠️ Frame read failed ({failure_count+1}/{max_retries}). Retrying...")

        failure_count += 1

        time.sleep(1)

        if failure_count >= max_retries:

            print("❌ Too many failures. Exiting.")

            break

        continue
 
    # Reset failure count after a successful read

    failure_count = 0
 
    # Run YOLO face detection

    try:

        results = model(frame)

    except Exception as e:

        print(f"❌ YOLO model failed: {e}")

        break
 
    # Count and draw bounding boxes

    count = 0

    for result in results:

        count += len(result.boxes)

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, "Face", (x1, y1 - 10),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 
    # Speak and send to Arduino if face count changed

    if count != previous_count:

        speak(f"{count} face{'s' if count != 1 else ''} detected")

        previous_count = count
 
        if arduino:

            try:

                arduino.write(f"{count}\n".encode())

                print(f"📤 Sent to Arduino: {count}")

            except Exception as e:

                print(f"⚠️ Error sending to Arduino: {e}")
 
    # Show video

    cv2.imshow("YOLO Face Detection", frame)
 
    # Exit on 'q'

    if cv2.waitKey(1) & 0xFF == ord('q'):

        print("🛑 Quit command received.")

        break
 
# === Cleanup ===

cap.release()

cv2.destroyAllWindows()

if arduino:

    arduino.close()

    print("🔌 Serial connection closed.")

 