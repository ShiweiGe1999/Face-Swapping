import cv2
import numpy as np

class FaceSegmentation:
    
    def __init__(self, landmarks1, polygon_first_img) -> None:
        self.landmarks1 = landmarks1
        self.polygon_first_img = polygon_first_img
        self.rect1 = FaceSegmentation.get_bounding_rect(polygon_first_img)
        self.triangulation()
        self.get_index()
    
    @staticmethod
    def get_bounding_rect(polygon):
        return cv2.boundingRect(polygon)

    def triangulation(self):
        '''
        returns a list of triangles with flat np array (shape of (6, ))
        '''
        
        subdiv = cv2.Subdiv2D(self.rect1)
        
        subdiv.insert(self.landmarks1)
        triangles_flat = np.array(subdiv.getTriangleList(), np.int32)
        self.triangles_flat = triangles_flat
        
    def draw_rect(self, img):
        x, y, w, h = self.rect1
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))

    def draw_triangles_flat(img, triangles_flat):
        for t in triangles_flat:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            
            cv2.line(img, pt1, pt2, (0, 0, 255), 2)
            cv2.line(img, pt2, pt3, (0, 0, 255), 2)
            cv2.line(img, pt3, pt1, (0, 0, 255), 2)
            
    def draw_triangles(self, img, triangles):
        '''
        Need to specify where to draw the triangle (list of 3 points)
        Imput image or output image
        '''
        for t in triangles:
            pt1 = t[0]
            pt2 = t[1]
            pt3 = t[2]

            cv2.line(img, pt1, pt2, (0, 0, 255), 2)
            cv2.line(img, pt2, pt3, (0, 0, 255), 2)
            cv2.line(img, pt3, pt1, (0, 0, 255), 2)
            
    def get_index(self):
        points = np.array(self.landmarks1, np.int32)
        
        indexes_triangles = []
        for t in self.triangles_flat:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            
            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = self.extract_index_nparray(index_pt1)
            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = self.extract_index_nparray(index_pt2)
            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = self.extract_index_nparray(index_pt3)
            
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)
            
        self.indexes_triangles = indexes_triangles

    def extract_index_nparray(self, nparray):
        index = None
        for num in nparray[0]:
            index = num
            break
        return index

    def get_triangle_locs(self, landmarks, indexes_triangles):
        triangles = []
        for triangle_idx in indexes_triangles:
        

            pt1 = landmarks[triangle_idx[0]]
            pt2 = landmarks[triangle_idx[1]]
            pt3 = landmarks[triangle_idx[2]]
            
            triangles.append((pt1, pt2, pt3))
            
        return triangles
    
    def get_input_triangle_locs(self):
        return self.get_triangle_locs(self.landmarks1, self.indexes_triangles)
    
    def get_output_triangle_locs(self, landmarks2):
        return self.get_triangle_locs(landmarks2, self.indexes_triangles)

    @staticmethod
    def get_corrisbonding_triangles(landmarks1, polygon_first_img, landmarks2, img1=None, img2=None, draw_triangles=False, draw_rect=False):
        face_segmentation = FaceSegmentation(landmarks1, polygon_first_img)
        
        triangles1 = face_segmentation.get_input_triangle_locs()
        triangles2 = face_segmentation.get_output_triangle_locs(landmarks2)
        
        if draw_triangles:
            if img1 is None or img2 is None:
                raise RuntimeError('Please specify image to draw triangles')
            
            face_segmentation.draw_triangles(img1, triangles1)
            face_segmentation.draw_triangles(img2, triangles2)
            
        if draw_rect:
            if img1 is None:
                raise RuntimeError('Please specify image to draw triangles')
            
            face_segmentation.draw_rect(img1)
            
        return triangles1, triangles2
            

            
    
    
