import cv2
import time
from control import Control
from hand_gesture import MPHandGesture

class Camera:
    
    def __init__(self,  mp: MPHandGesture, device: int=0, width:int=960, height:int=540,) -> None:
        self.device = device
        self.cap_width = width
        self.cap_height = height
        self.mp = mp
        
        self.cTime = 0
        self.pTime = 0
        
    def __draw_info(self, image, fps):
        cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def __cal_fps(self) -> int:
        self.cTime = time.time()
        fps = 1/(self.cTime-self.pTime)
        self.pTime = self.cTime
        return int(fps)
    
    def start_capture(self, draw_rect: bool=True):
        
        cap = cv2.VideoCapture(self.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_height)
        
        while Control.getInstance().start_capture:
            
            fps = self.__cal_fps()
            
            Control.getInstance().on_key_press()
            
            ret, image = cap.read()
            if not ret:
                break
            image = cv2.flip(image, 1)
            debug_image = image.copy()
            
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp.detect(image)
            
            debug_image = self.mp.draw2image(debug_image, results)
            
            debug_image = self.__draw_info(debug_image, fps)
            cv2.imshow('Hand Gesture Recognition', debug_image)
            
            
        cap.release()
        cv2.destroyAllWindows()
        
        
        
        