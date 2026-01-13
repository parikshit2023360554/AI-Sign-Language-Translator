import os
import cv2
import mediapipe as mp
import csv
import copy

def collect_data(class_name, sample_size=500):
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    
    # Setup data file
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    file_path = os.path.join(data_dir, 'hand_landmarks.csv')
    
    # write header if file doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 21 landmarks * 3 coords (x,y,z) = 63 columns + label
            header = ['label']
            for i in range(21):
                header.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
            writer.writerow(header)

    print(f"Collecting data for class: '{class_name}'. Press 's' to start/pause saving, 'q' to quit.")
    
    counter = 0
    saving = False
    
    while cap.isOpened() and counter < sample_size:
        success, image = cap.read()
        if not success:
            continue

        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                if saving:
                    # keypoints extraction
                    row = [class_name]
                    for landmark in hand_landmarks.landmark:
                        row.extend([landmark.x, landmark.y, landmark.z])
                    
                    with open(file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    
                    counter += 1
                    print(f"Saved sample {counter}/{sample_size}")

        cv2.putText(image, f"Class: {class_name} | Saved: {counter}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if saving:
            cv2.putText(image, "SAVING...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Press 's' to start", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Data Collection', image)
        
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            saving = not saving

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # You can change input here to collect different classes
    label = input("Enter the label name (e.g. 'A', 'Hello'): ")
    collect_data(label)
