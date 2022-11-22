import numpy as np
import cv2
from FaceSegmentation import FaceSegmentation


class SegmentSwapper:

    def __init__(self, img1, img2, triangles1, triangles2) -> None:
        # these two are lists
        self.img1 = img1
        self.img2 = img2
        self.triangles1 = triangles1
        self.triangles2 = triangles2


    def get_points_in_rect(self, triangle, rect):
        pt1, pt2, pt3 = triangle
        x, y, w, h = rect
        points = np.array([
            [pt1[0] - x, pt1[1] - y],
            [pt2[0] - x, pt2[1] - y],
            [pt3[0] - x, pt3[1] - y]
        ], np.int32)

        return points

    def get_cropped_rect(self, img, rect):
        x, y, w, h = rect
        return img[y: y + h, x: x + w]

    def swap_triangles(self):
        img2_new_face = np.zeros_like(self.img2)
        
        for triangle1, triangle2 in zip(self.triangles1, self.triangles2):

            triangle1_nparr = np.array(triangle1)
            rect1 = FaceSegmentation.get_bounding_rect(triangle1_nparr)
            cropped_triangle1 = self.get_cropped_rect(self.img1, rect1)
            _, _, w, h = rect1
            cropped_tr1_mask = np.zeros((h, w), np.uint8)
            points1 = self.get_points_in_rect(triangle1, rect1)

            # cv2.fillConvexPoly(cropped_tr1_mask, points1, 255)
            # cropped_triangle1 = cv2.bitwise_and(
            #     cropped_triangle1, cropped_triangle1, mask=cropped_tr1_mask)
            cv2.fillConvexPoly(cropped_tr1_mask, points1, 255)
            # cv2.imshow('c1', cropped_triangle1)

            triangle2_nparr = np.array(triangle2)
            rect2 = FaceSegmentation.get_bounding_rect(triangle2_nparr)
            # cropped_triangle2 = self.get_cropped_rect(self.img2, rect2)
            x, y, w, h = rect2
            cropped_tr2_mask = np.zeros((h, w), np.uint8)
            points2 = self.get_points_in_rect(triangle2, rect2)

            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
            # cropped_triangle2 = cv2.bitwise_and(
            #     cropped_triangle2, cropped_triangle2, mask=cropped_tr2_mask)

            # cv2.imshow('c2', cropped_triangle2)

            # wrapping start
            points1 = np.float32(points1)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points1, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle1, M, (w, h))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
            
            
            img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
            img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
            _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

            img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
            img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
            
        # face swapped
        # img2_face_mask = np.zeros_like()
        
        return img2_new_face

    @staticmethod
    def get_new_face(img1, img2, triangles1, triangles2):
        face_swapper = SegmentSwapper(img1, img2, triangles1, triangles2)
        return face_swapper.swap_triangles()
