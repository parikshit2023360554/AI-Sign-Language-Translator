import mediapipe as mp
print("Imported mediapipe successfully.")

try:
    print("Initializing mp.solutions.hands...")
    mp_hands = mp.solutions.hands
    
    print("Creating Hands object...")
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("Hands object created successfully.")
    
    hands.close()
    print("Hands object closed.")

except Exception as e:
    print(f"An error occurred: {e}")

print("MediaPipe test complete.")
