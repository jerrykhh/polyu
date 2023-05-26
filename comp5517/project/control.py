
import cv2
import csv

class Singleton:
    __instance = None

    @classmethod
    def getInstance(cls):
        if cls is Singleton:
            return None
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance
    
class Control(Singleton):
    
    def __init__(self) -> None:
        self.mode = 0
        self.classes = None
        self.start_capture = True
        with open('model/keypoint_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]
        
    
    def on_key_press(self):
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            self.start_capture = False
        else:
            self.select_mode(key)
    
    def __update_classes(self, key):
        
        if self.mode == 1 and key >= 65 and key < 91:
            self.classes = key-65
    
    def select_mode(self, key):
        
        self.__update_classes(key)
        
        if key == 47: # /
            if self.mode == 1:
                print("Change to mode 0")
                self.mode = 0
                self.classes = None
            else:
                print("Change to mode 1")
                self.mode = 1
                self.__update_classes(key)
        
        