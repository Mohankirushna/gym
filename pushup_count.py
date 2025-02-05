import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize Video Capture
cap = cv2.VideoCapture('/Users/mohankirushna.r/Downloads/videoplayback (3).mp4')  # Replace with your video path

# Variables to count push-ups
pushup_count = 0
is_pushing_up = False
threshold = 0.4  # Threshold to determine push-up position

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a better viewing angle
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)

    # Convert frame back to BGR for rendering
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Check if pose landmarks were detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get necessary landmarks (e.g., left elbow, right elbow, left shoulder, right shoulder)
        landmarks = results.pose_landmarks.landmark
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Define position conditions for push-ups (elbows bending)
        if left_elbow.y < left_shoulder.y and right_elbow.y < right_shoulder.y:  # Elbows bend for push-up
            is_pushing_up = True
        elif left_elbow.y > left_shoulder.y and right_elbow.y > right_shoulder.y and is_pushing_up:
            pushup_count += 1
            is_pushing_up = False

    # Display the frame
    cv2.putText(frame, f'Push-Ups: {pushup_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Push-Up Counter", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
