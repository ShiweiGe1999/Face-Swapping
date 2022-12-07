import cv2
from absl import app, flags
from FaceDetector import FaceDetector
from FaceSegmentation import FaceSegmentation
from SegmentSwapping import SegmentSwapper
from FaceSwapping import FaceSwaper
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('i', None, 'Input Image (first image path)')
flags.DEFINE_string('o', None, 'Swap the face to (second image path)')
flags.DEFINE_string('video', None, "Swap the face to the video")

class Main:

    @staticmethod
    def main(argv):
        del argv
        if FLAGS.i == None or (FLAGS.o == None and FLAGS.video == None):
            raise RuntimeError('Images or videos are not correctly specified!')
        
        Main.run(FLAGS.i, FLAGS.o, FLAGS.video)

    @staticmethod
    def run(img_path_from, img_path_to, video_path_to):
        if img_path_to:
            Main.run_pipeline(img_path_from, img_path_to)
        else:
            Main.run_pipeline_video(img_path_from, video_path_to)

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
        # cv2.imshow("img1", img1)
        # cv2.imshow("img2", img2)
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break
        
    @staticmethod
    def run_pipeline_video(img_path_from, video_path_to):
        # capture the video
        cap = cv2.VideoCapture(video_path_to)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter('build/output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        # Step 1: Detect for face and get landmarks from the first image
        img1, landmarks1, polygon1, _, _ = FaceDetector.get_img_landmarks_polygon_mask_and_face(
            img_path_from)
             
        while True:
            check, img2 = cap.read()
            if not check:
                break
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)      
            # Step 2: Detect for face and get landmarks from the second image
            face_detector_2 = FaceDetector(img2,img2_gray)
            landmarks2 = face_detector_2.get_landmarks()[0]
            points = np.array(landmarks2, np.int32)
            polygon2 = face_detector_2.get_convex_polygon(points)
            
            # Step 3: Find the corrisbonding triangles between two images 
            triangles1, triangles2 = FaceSegmentation.get_corrisbonding_triangles(landmarks1, polygon1, landmarks2)
            
            # Step 4: Generate the new face from first image
            new_face = SegmentSwapper.get_new_face(img1, img2, triangles1, triangles2)
            
            # Step 5: Put the generated face area to the second image
            result = FaceSwaper.get_swapped_res(img2, img2_gray, polygon2, new_face)
            cv2.imshow("Result", result)
            out.write(result)
            key = cv2.waitKey(1)
            if key == 27:
                break

if __name__ == '__main__':
    app.run(Main.main)

    
