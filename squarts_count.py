import cv2
import mediapipe as mp
import numpy as np
import sys

def findAngle(a, b, c, minVis=0.8):
    if a.visibility > minVis and b.visibility > minVis and c.visibility > minVis:
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])

        angle = np.arccos((np.dot(ba, bc)) / (np.linalg.norm(ba)
                                              * np.linalg.norm(bc))) * (180 / np.pi)

        if angle > 180:
            return 360 - angle
        else:
            return angle
    else:
        return -1


def legState(angle):
    if angle < 0:
        return 0  # Joint is not being picked up
    elif angle < 105:
        return 1  # Squat range
    elif angle < 150:
        return 2  # Transition range
    else:
        return 3  # Upright range


if __name__ == "__main__":

    # Init mediapipe drawing and pose
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Init Video Feed
    cap = cv2.VideoCapture("/Users/mohankirushna.r/Downloads/videoplayback (3).mp4")  # Replace with your video path

    if not cap.isOpened():
        print("Error: Video file not loaded.")
        sys.exit()

    while cap.read()[1] is None:
        print("Waiting for Video")

    # Main Detection Loop
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        # Initialize Reps and Body State
        repCount = 0
        lastState = 9

        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                print('Error: Image not found or could not be loaded.')
                break
            else:
                frame = cv2.resize(frame, (1024, 600))

            if ret == True:
                try:
                    # Convert frame to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame.flags.writeable = False

                    # Detect Pose Landmarks
                    lm = pose.process(frame).pose_landmarks
                    lm_arr = lm.landmark
                except:
                    print("Please Step Into Frame")
                    cv2.imshow("Squat Rep Counter", frame)
                    cv2.waitKey(1)
                    continue

                # Allow write, convert back to BGR
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Draw overlay with parameters:
                mp_drawing.draw_landmarks(frame, lm, mp_pose.POSE_CONNECTIONS, 
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

                # Calculate Angle
                rAngle = findAngle(lm_arr[24], lm_arr[26], lm_arr[28])
                lAngle = findAngle(lm_arr[23], lm_arr[25], lm_arr[27])

                # Calculate state
                rState = legState(rAngle)
                lState = legState(lAngle)
                state = rState * lState

                # Final state is product of two leg states
                if state == 0:  # One or both legs not detected
                    if rState == 0:
                        print("Right Leg Not Detected")
                    if lState == 0:
                        print("Left Leg Not Detected")
                elif state % 2 == 0 or rState != lState:  # One or both legs still transitioning
                    if lastState == 1:
                        if lState == 2 or lState == 1:
                            print("Fully extend left leg")
                        if rState == 2 or lState == 1:
                            print("Fully extend right leg")
                    else:
                        if lState == 2 or lState == 3:
                            print("Fully retract left leg")
                        if rState == 2 or lState == 3:
                            print("Fully retract right leg")
                else:
                    if state == 1 or state == 9:
                        if lastState != state:
                            lastState = state
                            if lastState == 1:
                                print("GOOD!")
                                repCount += 1

                print("Squats: " + str(repCount))

                cv2.imshow("Squat Rep Counter", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
