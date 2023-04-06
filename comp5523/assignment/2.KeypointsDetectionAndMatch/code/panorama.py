import cv2
import numpy as np
from tqdm import tqdm
from sift import keypoint_match, draw_match, transform
from utils import read_video_frames, write_and_show, destroyAllWindows, imshow

video_name = 'image/winter_day.mov'
images, fps = read_video_frames(video_name)
n_image = len(images)

h, w = images[0].shape[:2]
H, W = h, w*5
panorama = np.zeros([H,W,3])
# TODO: init panorama
h_start = H
w_start = W-w
panorama[:, w_start:w_start+w, :] = images[0]

trans_sum = np.zeros([H,W,3])
cnt = np.ones([H,W,1])*1e-10
for img in tqdm(images[::4], 'processing'):
    # TODO: stitch img to panorama one by one
    keypoints1, keypoints2, match = keypoint_match(panorama, img, max_n_match=1000, draw=False)
    keypoints1 = np.array([keypoints1[m.queryIdx].pt for m in match])
    keypoints2 = np.array([keypoints2[m.trainIdx].pt for m in match])
    aligned_img = transform(img, keypoints2, keypoints1, H, W)
    
    # combine
    trans_sum += aligned_img
    cnt += (aligned_img != 0).any(2, keepdims=True)
    panorama = trans_sum/cnt
    
    # show
    imshow('results/2_panorama.jpg', panorama)


# panorama = algined.mean(0)
write_and_show('results/2_panorama.jpg', panorama)

destroyAllWindows();
