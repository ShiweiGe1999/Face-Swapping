import cv2
from absl import app, flags
from FaceDetector import FaceDetector
from FaceSegmentation import FaceSegmentation
from SegmentSwapping import SegmentSwapper
from FaceSwapping import FaceSwaper

FLAGS = flags.FLAGS
flags.DEFINE_string('i', None, 'Input Image (first image path)')
flags.DEFINE_string('o', None, 'Swap the face to (second image path)')

class Main:

    @staticmethod
    def main(argv):
        del argv
        if FLAGS.i == None or FLAGS.o == None:
            raise RuntimeError('Images are not correctly specified!')
        
        Main.run(FLAGS.i, FLAGS.o)

    @staticmethod
    def run(img_path_from, img_path_to):
        Main.run_pipeline(img_path_from, img_path_to)

    @staticmethod
    def run_pipeline(img_path_from, img_path_to):
        # Step 1: Detect for face and get landmarks from the first image
        img1, landmarks1, polygon1, _, _ = FaceDetector.get_img_landmarks_polygon_mask_and_face(
            img_path_from)
        
        # Step 2: Detect for face and get landmarks from the second image
        img2, img2_gray, landmarks2, polygon2 = FaceDetector.get_img_img2_gray_landmakrs_polygon(
            img_path_to)
        
        # Step 3: Find the corrisbonding triangles between two images 
        triangles1, triangles2 = FaceSegmentation.get_corrisbonding_triangles(landmarks1, polygon1, landmarks2)
        
        # Step 4: Generate the new face from first image
        new_face = SegmentSwapper.get_new_face(img1, img2, triangles1, triangles2)
        
        # Step 5: Put the generated face area to the second image
        result = FaceSwaper.get_swapped_res(img2, img2_gray, polygon2, new_face)

        cv2.imshow("Result", result)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(Main.main)

    
