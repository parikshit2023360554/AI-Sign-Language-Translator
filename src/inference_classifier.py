import cv2
import mediapipe as mp
import pickle
import numpy as np

def main():
    # 1. Load the trained model
    try:
        model_dict = pickle.load(open('./models/pixel_model.p', 'rb'))
        model = model_dict['model']
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model file not found. Run 'src/train_pixel_classifier.py' first.")
        return

    # 2. Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    # 3. Label Map
    # Sign Language MNIST mapping (J=9 and Z=25 are usually missing/excluded in static datasets)
    labels_dict = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
        18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
    }
    
    # 4. Open Webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    print("Starting Inference... Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            continue

        H, W, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Display feedback
        prediction_text = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # --- Get Bounding Box (ROI) ---
                x_List = []
                y_List = []
                for lm in hand_landmarks.landmark:
                    x_List.append(lm.x)
                    y_List.append(lm.y)

                min_x = int(min(x_List) * W) - 20
                max_x = int(max(x_List) * W) + 20
                min_y = int(min(y_List) * H) - 20
                max_y = int(max(y_List) * H) + 20

                # Clip to frame boundaries
                min_x = max(0, min_x)
                min_y = max(0, min_y)
                max_x = min(W, max_x)
                max_y = min(H, max_y)

                # Ensure valid ROI
                if max_x - min_x > 0 and max_y - min_y > 0:
                    # Crop
                    hand_img = frame[min_y:max_y, min_x:max_x]
                    
                    # --- Preprocessing for Model ---
                    try:
                        # 1. Convert to Grayscale
                        hand_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                        
                        # 2. Resize to 28x28 (Model Requirement)
                        hand_resized = cv2.resize(hand_gray, (28, 28))
                        
                        # 3. Flatten (1D array)
                        hand_flat = hand_resized.flatten().reshape(1, -1)
                        
                        # 4. Normalize (0-1)
                        hand_norm = hand_flat / 255.0

                        # --- Predict ---
                        prediction = model.predict(hand_norm)
                        predicted_index = int(prediction[0])
                        prediction_char = labels_dict.get(predicted_index, "?")

                        # Draw Box and Text
                        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 0), 4)
                        cv2.putText(frame, prediction_char, (min_x, min_y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(frame, prediction_char, (min_x, min_y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)

                    except Exception as e:
                        print(f"Prediction Error: {e}")

        cv2.imshow('Sign Language Translator', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
