import cv2
import numpy as np

class FaceSwaper:
    def __init__(self, img2, img2_gray, polygon2, img2_new_face) -> None:
        self.img2 = img2
        self.img2_gray = img2_gray
        self.polygon2 = polygon2
        self.img2_new_face = img2_new_face
    
    def swap_face(self):
        img2_face_mask = np.zeros_like(self.img2_gray)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, self.polygon2, 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)
        
        img2_head_noface = cv2.bitwise_and(self.img2, self.img2, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, self.img2_new_face)
        (x, y, w, h) = cv2.boundingRect(self.polygon2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
        seamlessclone = cv2.seamlessClone(result, self.img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
        return seamlessclone
    
    @staticmethod
    def get_swapped_res(img2, img2_gray, polygon2, img2_new_face):
        face_swaper = FaceSwaper(img2, img2_gray, polygon2, img2_new_face)
        return face_swaper.swap_face()
