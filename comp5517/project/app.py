from camera import Camera
from hand_gesture import MPHandGesture
import argparse

def init():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--draw_rect", help='Draw Rect if detected', type=bool, default=True)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    
    args = parser.parse_args()
    main(args)
    

def main(args):
    
    Camera(mp=MPHandGesture(min_detection_confidence=args.min_detection_confidence), 
           device=args.device, 
           width=args.width, 
           height=args.height).start_capture(draw_rect=args.draw_rect)
    

if __name__ == "__main__":
    init()