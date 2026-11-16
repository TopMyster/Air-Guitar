from unicodedata import category

import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionTaskRunningMode = mp.tasks.vision.RunningMode

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
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            # print(result.hand_landmarks)
            binary_result = len(result.hand_landmarks)
            if binary_result == 1:
                print("Hand is on neck")
            elif binary_result == 2:
                print("Strumming")
            if result.hand_landmarks:
                print(result.handedness[0][0].category_name)

            cv2.imshow("Air Guitar", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
