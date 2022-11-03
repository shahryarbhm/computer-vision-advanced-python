import cv2
import mediapipe as mp
import time


class PoseDetector:
    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mode = static_image_mode
        self.modelComplexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = min_detection_confidence
        self.trackCon = min_tracking_confidence
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode,
            self.modelComplexity,
            self.smooth_landmarks,
            self.enable_segmentation,
            self.smooth_segmentation,
            self.detectionCon,
            self.trackCon,
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img,
                    self.results.pose_landmarks,
                    connections=self.mpPose.POSE_CONNECTIONS,
                )
        return img

    def getLandmarks(self, img, draw=True):
        lmLst = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmLst.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmLst


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getLandmarks(img, draw=False)
        print(lmList)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(
            img,
            "Fps: {}".format(str(int(fps))),
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 255),
            3,
        )
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
