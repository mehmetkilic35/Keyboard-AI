import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

box_keys = [
    ("A", "S", "H"),
    ("E", "D", "K"),
    ("I", "B", "C"),
    ("N", "C", "F"),
    ("R", "U", "Z"),
    ("L", "G", "O"),
    ("K", "Y", "P"),
    ("T", "O", "G"),
    ("M", "S", "Q"),
    ("U", "P", "X", "W"),
    (" ",),  # Space key
    ("DEL",),  # Delete key
    ("CLR",)  # Clear key
]

boxes = [(i * 50, 50, 50, 70) for i in range(len(box_keys))]  # Boxes are arranged horizontally

waiting_time = 0.7  # Configurable waiting time

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    last_box = None
    last_touch_time = None
    current_key_index = 0
    text = ""

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0])
                
                in_any_box = False
                for i, box in enumerate(boxes):
                    in_box = box[0] < x < box[0] + box[2] and box[1] < y < box[1] + box[3]
                    if in_box:
                        in_any_box = True
                        cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), -1)
                        if last_box != i:
                            last_touch_time = time.time()
                            last_box = i
                            current_key_index = 0
                        elif time.time() - last_touch_time > waiting_time:
                            cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), -1)
                            if box_keys[i][0] == " ":
                                text += " "
                            elif box_keys[i][0] == "DEL":
                                text = text[:-1]
                            elif box_keys[i][0] == "CLR":
                                text = ""
                            else:
                                current_key_index = (current_key_index + 1) % len(box_keys[i])
                            last_touch_time = time.time()
                    elif last_box == i:  # The finger just left this box
                        if box_keys[i][0] not in {" ", "DEL", "CLR"}:  # Do not add Space, Delete, and Clear keys to the text
                            text += box_keys[i][current_key_index]  # Add the current key to the text
                        last_box = None

                if not in_any_box and last_box is not None:
                    last_box = None

                # Draw boxes and label them
                for i, box in enumerate(boxes):
                    cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 2)
                    if last_box == i and box_keys[i][0] not in {" ", "DEL", "CLR"}:
                        cv2.putText(image, box_keys[i][current_key_index], (box[0]+20, box[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        for j, key in enumerate(box_keys[i]):
                            cv2.putText(image, key, (box[0]+5, box[1]+15+j*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Display text in the middle of the screen
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
        text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][1]
        text_x = (image.shape[1] - text_width) // 2
        text_y = (image.shape[0] + text_height) // 2  # Changed to display text at bottom
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)

        cv2.imshow('Image', image)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object
    cap.release()

    # Destroy all windows
    cv2.destroyAllWindows()
