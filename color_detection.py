import cv2
import numpy as np
import os

# Video file name
video_path = "my_video.mp4"

# Check if the file exists
if not os.path.exists(video_path):
    print(f"❌ Video file not found: {video_path}")
    exit()

# Load the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("⚠️ Failed to open the video.")
    exit()

print("✅ Video opened successfully.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("📛 End of video or failed to read frame.")
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define wider blue color range
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask for blue areas
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours (edges) in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected blue areas
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Lowered threshold to detect small objects
            x, y, w, h = cv2.boundingRect(cnt)
            # Draw sky blue rectangle (B, G, R)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # Show the result
    cv2.imshow("Blue Object Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("👋 Exiting video playback.")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()