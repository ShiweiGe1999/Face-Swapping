import cv2
import numpy as np
import dlib
from ImageHelper import ImageHelper


class FaceDetector:

    def __init__(self, img, img_gray) -> None:
        self.img = img
        self.img_gray = img_gray
        self.mask = ImageHelper.create_mask(img_gray)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat")

    def get_landmarks(self):
        faces = self.detector(self.img_gray)

        all_landmarks = []

        for face in faces:
            landmarks = self.predictor(self.img_gray, face)
            landmarks_points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))

            all_landmarks.append(landmarks_points)

        return all_landmarks

    def plot_landmarks(self, landmarks):
        for x, y in landmarks:
            cv2.circle(self.img, (x, y), 3, (0, 0, 255), -1)
        
    def get_convex_polygon(self, landmarks):
        return cv2.convexHull(landmarks)
    
    def draw_convex_polygon(self, polygon):
        cv2.polylines(self.img, [polygon], True, (255, 0, 0), 3)
        
    def fill_convex_polygon(self, polygon, color=(255, 255, 255)):
        cv2.fillConvexPoly(self.mask, polygon, color)
        
    def extract_with_mask(self):
        return cv2.bitwise_and(self.img, self.img, mask=self.mask)
    
    @staticmethod
    def get_img_landmarks_polygon_mask_and_face(img_path, plot_landmarks_on_img=False, draw_convex_polygon_on_img=False):
        img = ImageHelper.read_img(img_path)
        img_gray = ImageHelper.get_img_grey(img)
        
        face_detector = FaceDetector(img, img_gray)
        all_landmarks = face_detector.get_landmarks()
        landmarks = all_landmarks[0]
        points = np.array(landmarks, np.int32)
        polygon = face_detector.get_convex_polygon(points)
        face_detector.fill_convex_polygon(polygon)
        face_only = face_detector.extract_with_mask()
        
        if plot_landmarks_on_img:
            face_detector.plot_landmarks(landmarks)
            
        if draw_convex_polygon_on_img:
            face_detector.draw_convex_polygon(polygon)
        
        return (img, landmarks, polygon, face_detector.mask, face_only)
    
    @staticmethod
    def get_img_img2_gray_landmakrs_polygon(img_path, plot_landmarks_on_img=False):
        img = ImageHelper.read_img(img_path)
        img_gray = ImageHelper.get_img_grey(img)
        face_detector = FaceDetector(img, img_gray)
        all_landmarks = face_detector.get_landmarks()
        landmarks = all_landmarks[0]
        points = np.array(landmarks, np.int32)
        polygon = face_detector.get_convex_polygon(points)
        
        if plot_landmarks_on_img:
            face_detector.plot_landmarks(landmarks)
        
        return (img, img_gray, landmarks, polygon)

def test():
    
    img, landmarks, polygon, mask, face_only = FaceDetector.get_img_landmarks_polygon_mask_and_face(
        'images/joe.jpg', plot_landmarks_on_img=True, draw_convex_polygon_on_img=True)
    
    cv2.imshow("Result", face_only)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()
