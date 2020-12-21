import os
import imutils
import cv2
from loguru import logger

from Motpy.MultiObjectTracker import MultiObjectTracker
from Motpy.Detection import Detection
from Motpy.testing_viz import draw_detection, draw_track

"""

    Example uses built-in camera (0) and baseline face detector from OpenCV (10 layer ResNet SSD)
    to present the library ability to track a face of the user

"""



class Detector(object):
    def __init__(self):
    
        self.net = net = cv2.dnn.readNetFromCaffe("SSD_files/MobileNetSSD_deploy.prototxt", "SSD_files/MobileNetSSD_deploy.caffemodel")

    def process(self, frame, conf_threshold=0.5):
        (h, w) = frame.shape[:2]
        #blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), [104, 117, 123], False, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        #print(detections.shape)
        # convert output from OpenCV detector to tracker expected format [xmin, ymin, xmax, ymax]
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                xmin = int(detections[0, 0, i, 3] * frame.shape[1])
                ymin = int(detections[0, 0, i, 4] * frame.shape[0])
                xmax = int(detections[0, 0, i, 5] * frame.shape[1])
                ymax = int(detections[0, 0, i, 6] * frame.shape[0])
                bboxes.append([xmin, ymin, xmax, ymax])

        return bboxes


def run():
    # prepare multi object tracker
    model_spec = {'order_pos': 1, 'dim_pos': 2,
                  'order_size': 0, 'dim_size': 2,
                  'q_var_pos': 5000., 'r_var_pos': 0.1}

    dt =  1/15.0  # assume 15 fps
    tracker = MultiObjectTracker(dt=dt, model_spec=model_spec)

    # open camera
    cap = cv2.VideoCapture('videos/1.mp4')
    updated_counter=0
    face_detector = Detector()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        frame = imutils.resize(frame, width=600)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # run face detector on current frame
        bboxes = face_detector.process(rgb)
        
        detections = [Detection(box=bbox) for bbox in bboxes]

        counter=updated_counter
        
        nb=tracker.cleanup_trackerss()
        
        _,updated_counter=tracker.step(detections,counter)
        
        

       
        tracks = tracker.active_tracks(min_steps_alive=3)
        
      
        
        # preview the boxes on frame
        for det in detections:
           draw_detection(frame, det)
        
        
        
        for track in tracks:
            draw_track(frame, track)
        
        cv2.imshow('frame', frame)

        # stop demo by pressing 'q'
        if cv2.waitKey(int(1000 * dt)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
