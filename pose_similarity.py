import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Thresholds for counting for each exercise
UP_THRESHOLD = 0.4
DOWN_THRESHOLD = 0.8
LEFT_TURN_THRESHOLD = 0.4  # Head turned to the left (relative x-position of the nose)
RIGHT_TURN_THRESHOLD = 0.6  # Head turned to the right
SHRUG_THRESHOLD = 0.3  # Shoulder shrug threshold (relative y-position of shoulders)

# Exercise states
exercise_counter = 0
arm_up = False
current_exercise = 0
total_exercises = 3
repetitions_per_exercise = 10
exercise_names = ["Arm Raise", "Head Turn", "Shoulder Shrugs"]
head_turn_state = None  # To track the direction of the last head turn

# OpenCV video capture
cap = cv2.VideoCapture(0)

def countdown(image):
    for i in range(3, 0, -1):
        cv2.putText(image, str(i), (int(image.shape[1]/2) - 50, int(image.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10, cv2.LINE_AA)
        cv2.imshow('Exercise', image)
        cv2.waitKey(1000)
    cv2.putText(image, "Start!", (int(image.shape[1]/2) - 100, int(image.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10, cv2.LINE_AA)
    cv2.imshow('Exercise', image)
    cv2.waitKey(1000)

# Exercise Logic for Arm Raise
def arm_raise_exercise(image, results):
    global arm_up, exercise_counter
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        
        if left_wrist_y < UP_THRESHOLD and right_wrist_y < UP_THRESHOLD:
            if not arm_up:
                arm_up = True
                print("Arms raised")
        
        if left_wrist_y > DOWN_THRESHOLD and right_wrist_y > DOWN_THRESHOLD:
            if arm_up:
                arm_up = False
                exercise_counter += 1
                print(f"Exercise count: {exercise_counter}")
    
    return image

# Exercise Logic for Head Turn (Alternating Left and Right)
def head_turn_exercise(image, results):
    global exercise_counter, head_turn_state
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        nose_x = landmarks[mp_pose.PoseLandmark.NOSE].x
        
        if nose_x < LEFT_TURN_THRESHOLD and head_turn_state != 'left':
            head_turn_state = 'left'
            exercise_counter += 1
            print(f"Head turned left. Count: {exercise_counter}")
        elif nose_x > RIGHT_TURN_THRESHOLD and head_turn_state != 'right':
            head_turn_state = 'right'
            exercise_counter += 1
            print(f"Head turned right. Count: {exercise_counter}")
    
    return image

# Exercise Logic for Shoulder Shrugs
def shoulder_shrug_exercise(image, results):
    global exercise_counter
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        
        # Check if both shoulders are raised (y-position decreases)
        if left_shoulder_y < SHRUG_THRESHOLD and right_shoulder_y < SHRUG_THRESHOLD:
            exercise_counter += 1
            print(f"Shoulder shrug. Count: {exercise_counter}")
    
    return image

# Display the "Well Done" message
def display_well_done(image):
    cv2.putText(image, "Well Done!", (int(image.shape[1]/2) - 200, int(image.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
    cv2.imshow('Exercise', image)
    cv2.waitKey(3000)

# Main loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Display the current exercise name
    cv2.putText(image, f"Exercise: {exercise_names[current_exercise]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Perform exercise-specific logic
    if current_exercise == 0:
        image = arm_raise_exercise(image, results)
    elif current_exercise == 1:
        image = head_turn_exercise(image, results)
    elif current_exercise == 2:
        image = shoulder_shrug_exercise(image, results)
    
    # Check if 10 repetitions have been completed
    if exercise_counter >= repetitions_per_exercise:
        current_exercise += 1
        exercise_counter = 0
        head_turn_state = None  # Reset head turn state for the next exercise
        if current_exercise >= total_exercises:
            display_well_done(image)
            break
        else:
            countdown(image)  # Start the next exercise with a countdown

    # Display the count
    cv2.putText(image, f"Count: {exercise_counter}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Draw the pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Display the image
    cv2.imshow('Exercise', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
