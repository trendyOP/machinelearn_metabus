import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['greetings', 'hello', 'meet', 'part', 'glad', 'worry', 'introduction', 'name', 'age', 'you', 'me', 'live', 'know', 'dont know', 'right', 'no', 'what', 'thanks', 'fine', 'want']
translations = {'greetings': 'greet', 'hello': 'hello', 'meet': 'meet', 'part': 'part', 'glad': 'glad', 'worry': 'worry', 'introduction': 'intro', 'name': 'name', 'age': 'age', 'you': 'u', 'me': 'me', 'live': 'live', 'know': 'know', 'dont know': 'dont know', 'right': 'right', 'no': 'no', 'what': 'what', 'thanks': 'thank', 'fine': 'fine', 'want': 'want'}
seq_length = 30
secs_for_action = 60

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(1)
created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []
        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 3))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z]
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
                    v = v2 - v1
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                    angle = np.arccos(np.einsum('nt,nt->n', v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :], v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                    angle = np.degrees(angle)
                    data.append(np.concatenate([joint.flatten(), angle]))
                    
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            
            cv2.putText(img, f'{action} ({int(time.time() - start_time)})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break
                
        data = np.array(data)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), data)

cap.release()
cv2.destroyAllWindows()
