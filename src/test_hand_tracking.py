import cv2
import mediapipe as mp
import time

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize Webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Starting Hand Tracking... Press 'q' to exit.")

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert the BGR image to RGB for MediaPipe
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and find hands
            results = hands.process(image_rgb)

            # Draw the annotations on the image
            image.flags.writeable = True
            # Convert back to BGR for OpenCV display
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 1. Draw landmarks and connections
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # 2. Print coordinates and label landmarks
                    h, w, c = image.shape
                    print(f"--- New Frame ---")
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        # Print to terminal
                        print(f"ID: {idx}, x: {landmark.x:.4f}, y: {landmark.y:.4f}, z: {landmark.z:.4f}")
                        
                        # Calculate pixel coordinates for drawing text
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        
                        # Draw index number on the hand
                        cv2.putText(
                            image, 
                            str(idx), 
                            (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (255, 0, 0), # Blue color
                            1, 
                            cv2.LINE_AA
                        )

            # Show the image
            cv2.imshow('MediaPipe Hand Tracking', image)

            # Exit condition
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Webcam released. Exiting.")

if __name__ == "__main__":
    main()
