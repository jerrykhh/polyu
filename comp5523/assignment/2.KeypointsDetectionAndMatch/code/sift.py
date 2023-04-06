import cv2
import numpy as np
from utils import imread, imshow, write_and_show, destroyAllWindows

def keypoint_match(img1, img2, max_n_match=100, draw=True):
    # make sure they are of dtype uint8
    img1 = np.uint8(img1)
    img2 = np.uint8(img2)

    # TODO: convert to grayscale by `cv2.cvtColor`
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    

    # TODO: detect keypoints and generate descriptor by `sift.detectAndCompute`
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, mask=None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, mask=None)


    # draw keypoints
    if draw:
        write_and_show('./results/1.2_img1_gray.jpg', img1_gray)
        write_and_show('./results/1.2_img2_gray.jpg', img2_gray)
        # TODO: draw keypoints on image1 and image2 by `cv2.drawKeypoints`
        
        img1_keypoints = cv2.drawKeypoints(
                image     = img1,
                keypoints = keypoints1,
                outImage  = None,
                flags     = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        img2_keypoints = cv2.drawKeypoints(
                image     = img2,
                keypoints = keypoints2,
                outImage  = None,
                flags     = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        write_and_show('results/1.3_img1_keypoints.jpg', img1_keypoints)
        write_and_show('results/1.3_img2_keypoints.jpg', img2_keypoints)


    # TODO: Knn match and Lowe's ratio test
    matcher = cv2.FlannBasedMatcher_create()
    match = matcher.knnMatch(
            queryDescriptors = descriptors1, 
            trainDescriptors = descriptors2,
            k=2)
    
    ratio = 0.7
    match = [m for m,n in match if m.distance < ratio*n.distance]



    # TODO: select best `max_n_match` matches
    match = sorted(match, key = lambda x:x.distance)
    match = match[:max_n_match]

    return keypoints1, keypoints2, match


def draw_match(img1, keypoints1, img2, keypoints2, match, savename):
    img1 = np.uint8(img1)
    img2 = np.uint8(img2)

    # TODO: draw matches by `cv2.drawMatches`
    match_draw = cv2.drawMatches(
        img1        = img1,
        keypoints1  = keypoints1,
        img2        = img2,
        keypoints2  = keypoints2,
        matches1to2 = match,
        outImg      = None,
        flags       = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    write_and_show(savename, match_draw)


def transform(img, img_kps, dst_kps, H, W):
    '''
    Transfrom img such `img_kps` are aligned with `dst_kps`.
    H: height of output image
    W: width of output image
    '''
    # TODO: get transform matrix by `cv2.findHomography`
    T, status = cv2.findHomography(
                        srcPoints = img_kps,
                        dstPoints = dst_kps,
                        method    = cv2.USAC_ACCURATE,
                        ransacReprojThreshold = 3)
    print(T)
    # TODO: apply transform by `cv2.warpPerspective`
    transformed = cv2.warpPerspective(
                    src   = img,
                    M     = T,
                    dsize = (W, H),
                    dst   = np.zeros_like(img, shape=(H,W)),
                    borderMode = cv2.BORDER_TRANSPARENT)


    return transformed


if __name__ == '__main__':
    ## read images
    img1 = imread('./image/left2.jpg')
    img2 = imread('./image/right2.jpg')

    ## find keypoints and matches
    keypoints1, keypoints2, match = keypoint_match(img1, img2, max_n_match=1000)

    draw_match(img1, keypoints1, img2, keypoints2,
               match, savename='results/1.4_match.jpg')

    # get all matched keypoints
    keypoints1 = np.array([keypoints1[m.queryIdx].pt for m in match])
    keypoints2 = np.array([keypoints2[m.trainIdx].pt for m in match])

    ## Align img2 to img1
    H, W = img1.shape[:2]
    W = W*2
    new_img2 = transform(img2, keypoints2, keypoints1, H, W)
    write_and_show('results/1.6_transformed.jpg', new_img2)

    # resize img1
    new_img1 = np.hstack([img1, np.zeros_like(img1)])
    # write_and_show('results/new_img1.jpg', new_img2)

    # TODO: average `new_img1` and `new_img2`
    cnt = np.zeros([H,W,1]) + 1e-10
    cnt += (new_img2 != 0).any(2, keepdims=True)
    cnt += (new_img1 != 0).any(2, keepdims=True)
    
    new_img1 = np.float32(new_img1)
    new_img2 = np.float32(new_img2)

    stack = (new_img2+new_img1)/cnt

    write_and_show('results/1.7_stack.jpg', stack)

    destroyAllWindows()
