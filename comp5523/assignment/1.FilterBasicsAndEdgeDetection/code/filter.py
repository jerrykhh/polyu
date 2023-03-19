from PIL import Image # pillow package
import numpy as np
from scipy import ndimage

def read_img_as_array(file) -> np.ndarray:
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)

def show_array_as_img(arr, rescale='minmax'):
    
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.show()

def rgb2gray(arr):
    R = arr[:, :, 0] # red channel
    G = arr[:, :, 1] # green channel
    B = arr[:, :, 2] # blue channel
    gray = 0.2989*R + 0.5870*G + 0.1140*B
    return gray

#########################################
## Please complete following functions ##
#########################################
def sharpen(img: np.ndarray, sigma, alpha: float):
    '''Sharpen the image. 'sigma' is the standard deviation of Gaussian filter. 'alpha' controls how much details to add.'''
    # TODO: Please complete this function.
    # your code here
    blurred = ndimage.gaussian_filter(img, sigma=sigma)
    detailed = np.float64(img) - np.float64(blurred)
    sharpened = np.clip(np.float64(img) + alpha * detailed, 0, 255)
    
    show_array_as_img(sharpened)
    return sharpened

def median_filter(img: np.ndarray, s):
    '''Perform median filter of size s x s to image 'arr', and return the filtered image.'''
    # TODO: Please complete this function.
    # your code here
    
    # return ndimage.median_filter(img, size=s)
    
    h, w, channels = img.shape
    result: np.ndarray = img
    padding = s // 2

    for channel in range(channels):
        data = img[:, :, channel]
        print("channel:",channel)
        
        flatten = []
        output = []
        output = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                for z in range(s):
                    if z + - padding < 0 or i + z - padding > h - 1:
                        for _ in range(s):
                            flatten.append(0)
                    else:
                        if j + z - padding < 0 or j + padding > w - 1:
                            flatten.append(0)
                        else:
                            for k in range(s):
                                flatten.append(data[i + z - padding][j + k - padding])

                flatten.sort()
                output[i][j] = flatten[len(flatten) // 2]
                flatten = []
                
        print(output.shape)
        result[:,:, channel] = output
        
    show_array_as_img(result)
    return result

if __name__ == '__main__':
    input_path = './data/rain.jpeg'
    img = read_img_as_array(input_path)
    show_array_as_img(img)
    
    #TODO: finish assignment Part I.
    
    save_array_as_img(sharpen(img.copy(), sigma=3, alpha=2.0),'./data/1.1_sharpened.jpg')
    save_array_as_img(median_filter(img.copy(), s=5), './data/1.2_derained.jpg')
    
    
