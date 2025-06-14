# 🏋️‍♂️ Exercise Pose Analysis & Rep Counter 🏃‍♀️
This repository contains Python scripts using MediaPipe Pose, OpenCV, and NumPy to monitor and count exercise repetitions from videos. The supported exercises include push-ups, squats, and rope jumping.

| Filename           | Exercise        | Description                                                     |
| ------------------ | --------------- | --------------------------------------------------------------- |
| `pushup_count.py`  | 💪 Push-ups     | Counts push-up reps by analyzing elbow and shoulder positions.  |
| `squarts_count.py` | 🏋️‍♀️ Squats   | Counts squat reps using knee joint angles and pose state logic. |
| `rope_count.py`    | 🪢 Rope Jumping | Tracks vertical hip centroid movement to count jumps.           |

🛠️ Requirements
Python 3.7+
OpenCV (opencv-python)
MediaPipe (mediapipe)
NumPy (numpy)

Install dependencies:
pip install opencv-python mediapipe numpy

▶️ Usage
Update the video path inside each script (file_name or hardcoded path), then run:
python pushup_count.py
python squarts_count.py
python rope_count.py
Each script processes the video and displays the real-time exercise count overlayed on the video.

📋 Details
pushup_count.py: Counts push-ups by detecting elbow bending relative to shoulders. 💪
squarts_count.py: Uses angle calculations at knees to identify squatting and count reps. 🏋️‍♀️
rope_count.py: Uses hip landmark vertical centroid tracking and smoothing buffers to count rope jumps. 🪢

⚙️ Customization
Modify video file paths to use your own videos. 🎥
Tune threshold values in each script to improve accuracy depending on your camera angle and exercise form. 🎯

💡 Notes
Works best with single person videos. 👤
Good lighting and clear side/front views improve detection. 💡
Real-time processing may require decent hardware. ⚡

📢 Important
Note: The camera angle for each activity (e.g., push-ups, skipping, etc.) should be fixed for optimal detection and accurate output. 🎥📐

📄 License
MIT License © Your Name
