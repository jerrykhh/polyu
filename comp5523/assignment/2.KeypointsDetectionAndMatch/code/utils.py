import cv2
import numpy as np
from tqdm import tqdm

def imread(filename):
    img = cv2.imread(filename)
    img = np.float32(img)
    return img

def imshow(title, img):
    if img.dtype == np.float32 or img.dtype == np.float64:
        cv2.imshow(title, img/255)
    else:
        cv2.imshow(title, img)
    cv2.waitKey(1)

def write_and_show(filename, img):
    success = cv2.imwrite(filename, img)
    assert success, f'failed to save "{filename}"'
    imshow(filename, img)

def destroyAllWindows(wait_key=True):
    if wait_key:
        print("press any key in OpenCV windows to continue...")
        cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def read_video_frames(video_name):
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(video_name)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
      raise ValueError(f"Cannot opening file {video_name}")

    # Read all frames
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(frame_count), f'read from "{video_name}"'):
        ret, frm = cap.read()
        if ret: frames.append(frm)

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps

def write_frames_to_video(video_name, frames, fps):
    # fps stands for 'frame per second'
    H, W, _ = frames[0].shape
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('a','v','c','1'), fps, (W,H))
    for frm in tqdm(frames, f'write to "{video_name}" with fps={fps}'):
        frm = np.uint8(frm)
        out.write(frm)
    out.release()
