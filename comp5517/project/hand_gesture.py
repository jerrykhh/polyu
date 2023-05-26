import mediapipe as mp
import numpy as np
import cv2
import itertools
import csv
import tensorflow as tf
from control import Control

class MPHandGesture:
    
    class KeyPointClassifier:
        def __init__(self, model_path:str="model/keypoint_classifier.tflite", num_threads:int=1) -> None:
            self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
        def __call__(self,landmark_list):
            input_details_tensor_index = self.input_details[0]['index']
            self.interpreter.set_tensor(
                input_details_tensor_index,
                np.array([landmark_list], dtype=np.float32))
            self.interpreter.invoke()

            output_details_tensor_index = self.output_details[0]['index']

            result = self.interpreter.get_tensor(output_details_tensor_index)

            return np.argmax(np.squeeze(result))
    
    def __init__(self, max_num_hands:int=1, static_image_mode:bool=True, min_detection_confidence:float=0.7) -> None:

        self.mpHands = mp.solutions.hands
        self.hands =  self.mpHands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )
        self.keypoint_classifier = self.KeyPointClassifier()
    
    def detect(self, image):
        return self.hands.process(image)
    
    def __get_bbox_rext(self, image, landmarks):
        
        h, w = image.shape[0], image.shape[1]
        landmark_array = np.empty((0, 2), int)
        
        landmark_points = []
        
        for landmark in landmarks.landmark:
            landmark_x = min(int(landmark.x * w), w - 1)
            landmark_y = min(int(landmark.y * h), h - 1)
            
            landmark_points.append([landmark_x, landmark_y])
            
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)
        
        x, y, w, h = cv2.boundingRect(landmark_array)
        return [x, y, x + w, y + h], landmark_points
    
    def __landmark_pre_process(self, landmark_list):
        temp_landmark_list = landmark_list.copy()

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        return temp_landmark_list
    
    def __logging_csv(self, landmark_list=None):
        
        mode = Control.getInstance().mode
        classes = Control.getInstance().classes
        
        if mode == 0:
            pass
        elif mode == 1 and landmark_list is not None and classes is not None:
            csv_path = 'model/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([classes, *landmark_list])
    
    def __draw_bounding_rect(self, image, bbox):
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                        (0, 0, 0), 1)

        return image
    
    def __draw_hand_info_text(self, image, bbox, handedness, hand_sign_id:int=None):
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[1] - 22),
                    (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]
        if hand_sign_id is not None:
            
            info_text = info_text + ':' + Control.getInstance().keypoint_classifier_labels[int(hand_sign_id)]
        cv2.putText(image, info_text, (bbox[0] + 5, bbox[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return image
    
    def __draw_info(self, image):
        
        mode = Control.getInstance().mode
        classes = Control.getInstance().classes
        

        if 1 == mode:
            cv2.putText(image, "MODE: Logging Key Point", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                    cv2.LINE_AA)
            
            if classes is not None:
                cv2.putText(image, "classes:" + str(Control.getInstance().keypoint_classifier_labels[int(classes)]), (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                            cv2.LINE_AA)
        return image
    
    
    def draw2image(self, image, detected_results, draw_rect:bool=True):
        draw_image = image.copy()
        
        if detected_results.multi_hand_landmarks is not None:
            
            for hand_landmarks, handedness in zip(detected_results.multi_hand_landmarks,
                                                    detected_results.multi_handedness):
                
                bbox_rect, landmark_list = self.__get_bbox_rext(image, hand_landmarks)
            
                pre_processed_landmark_list = self.__landmark_pre_process(landmark_list)

                self.__logging_csv(pre_processed_landmark_list)
                
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                
                # draw
                if draw_rect:
                    draw_image = self.__draw_bounding_rect(image=draw_image, bbox=bbox_rect)
                
                if Control.getInstance().mode != 1:
                    draw_image = self.__draw_hand_info_text(image=draw_image, bbox=bbox_rect, handedness=handedness, hand_sign_id = hand_sign_id)
                
                mp.solutions.drawing_utils.draw_landmarks(draw_image, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
                
        draw_image = self.__draw_info(draw_image)
        return draw_image