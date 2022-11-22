import cv2
import numpy as np

class ImageHelper:
  
    def __init__(self) -> None:
        raise RuntimeError('Cannot ceate ImageReader instance!')
    
    @staticmethod
    def read_img(path):
        img = cv2.imread(path)
        return img
    
    @staticmethod
    def get_img_gray(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray
    
    @staticmethod
    def create_mask(img_gray):
        mask = np.zeros_like(img_gray)
        return mask
    
    @staticmethod
    def save_img(filename, img):
        cv2.imwrite(filename, img)
  
  