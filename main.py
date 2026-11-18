from unicodedata import category
import cv2
import mediapipe as mp
import time
from numpy.random.mtrand import randint
from playsound3 import playsound
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionTaskRunningMode = mp.tasks.vision.RunningMode

data = pd.read_csv("chords.csv")

def main():
    print("Hello Guitar")
    cap = cv2.VideoCapture(0)

    cap.set(3, 1280)
    cap.set(4, 720)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        clasf = RandomForestClassifier(random_state=0)
        X = data.drop(columns=["chord"])
        y = data["chord"]
        clasf.fit(X,y)

        last_strum = 0
        cool_down = 1

        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            hand = result.hand_landmarks
            binary_result = len(result.hand_landmarks)
            if binary_result == 1:
                print("Hand is on neck")

            elif binary_result == 2:
                dx = hand[0][9].x-hand[0][0].x
                dy = hand[0][9].y-hand[0][0].y
                size = (dx**2 + dy**2) ** 0.5

                f1x = (hand[0][8].x-hand[0][0].x) / size
                f1y = (hand[0][8].y-hand[0][0].y) / size

                f2x = (hand[0][12].x-hand[0][0].x) / size
                f2y = (hand[0][12].y-hand[0][0].y) / size

                f3x = (hand[0][16].x-hand[0][0].x) / size
                f3y = (hand[0][16].y-hand[0][0].y) / size

                f4x = (hand[0][20].x-hand[0][0].x) / size
                f4y = (hand[0][20].y-hand[0][0].y) / size

                chord = clasf.predict([[f1x,f1y,f2x,f2y,f3x,f3y,f4x,f4y,]])[0]

                now = time.time()

                if (now - last_strum) > cool_down:
                    playsound(f"chords/{chord}.m4a", block=False)
                    last_strum = now

            cv2.imshow("Air Guitar", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
