import cv2
# from os import path
import os
import numpy as np
from tqdm import tqdm
from sift import draw_match, transform
from utils import (
    imshow, imread,
    write_and_show, destroyAllWindows,
    read_video_frames, write_frames_to_video
)

def orb_keypoint_match(img1, img2, max_n_match=100):
    # make sure they are of dtype uint8
    img1 = np.uint8(img1)
    img2 = np.uint8(img2)

    # TODO: convert to grayscale by `cv2.cvtColor`
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    # TODO: detect keypoints and generate descriptor by by 'orb.detectAndCompute', modify parameters for cv2.ORB_create for more stable results.
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2)
    keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

    # TODO: convert descriptors1, descriptors2 to np.float32

    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)

    # TODO: Knn match and Lowe's ratio test
    matcher = cv2.FlannBasedMatcher_create()
    match = matcher.knnMatch(
            queryDescriptors = descriptors1, 
            trainDescriptors = descriptors2
            ,k=2)
    
    ratio = 0.7
    match = [m for m,n in match if m.distance < ratio*n.distance]


    # TODO: select best `max_n_match` matches
    match = sorted(match, key = lambda x:x.distance)
    match = match[:max_n_match]

    return keypoints1, keypoints2, match


if __name__ == '__main__':
    # read in video
    video_name = 'image/rain2.mov'
    images, fps = read_video_frames(video_name)
    images = np.asarray(images)

    # get stabilized frames
    stabilized = []
    reference = images[0]
    H, W = reference.shape[:2]
    for img in tqdm(images[::2], 'processing'):
        ## TODO find keypoints and matches between each input img and the reference image
        ref_kps, img_kps, match = orb_keypoint_match(reference, img, max_n_match=1500)
    
        ref_kps = np.array([ref_kps[m.queryIdx].pt for m in match])
        img_kps = np.array([img_kps[m.trainIdx].pt for m in match])
        # TODO: align all frames to reference frame (images[0])
        trans = transform(img, img_kps, ref_kps, H, W )


        stabilized.append(trans)
        imshow('trans.jpg', trans)

    # write stabilized frames to a video
    write_frames_to_video('results/3.2_stabilized.mp4', stabilized, fps/2)

    # get rain free images
    stabilized_mean = np.mean(stabilized, 0)
    write_and_show('results/3.3_stabilized_mean.jpg', stabilized_mean)

    destroyAllWindows()
