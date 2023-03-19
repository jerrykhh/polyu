from PIL import Image, ImageDraw # pillow package
import numpy as np
from scipy import ndimage

def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)

def show_array_as_img(arr):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
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

def sobel(arr):
    '''Apply sobel operator on arr and return the result.'''
    # TODO: Please complete this function.
    # your code here
    
    # Define Sobel operators
    Gx_operator = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Gy_operator = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Compute image gradients in x and y direction
    Gx = ndimage.convolve(arr, Gx_operator)
    Gy = ndimage.convolve(arr, Gy_operator)
    
    Gx *= 255 / Gx.max()
    Gy *= 255 / Gy.max()

    # Compute the magnitude of the gradients
    G = np.sqrt(np.square(Gx) + np.square(Gy))

    return G, Gx, Gy

def nonmax_suppress(G, Gx, Gy):
    '''Suppress non-max value along direction perpendicular to the edge.'''
    assert G.shape == Gx.shape
    assert G.shape == Gy.shape
    # TODO: Please complete this function.
    # your code here
    
    # Initialize suppressed gradient
    suppressed_G = np.zeros_like(G)

     # Compute angle between gradient and x-axis
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta[theta < 0] += 180
    
    # Suppress non-maximum values
    for i in range(1, G.shape[0] - 1):
        for j in range(1, G.shape[1] - 1):
            # Get gradient values of neighboring pixels along the gradient direction
            if (0 <= theta[i, j] < 22.5) or (157.5 <= theta[i, j] <= 180) or (-22.5 <= theta[i, j] < 0) or (-180 <= theta[i, j] < -157.5):
                N1, N2 = G[i, j+1], G[i, j-1]
            elif (22.5 <= theta[i, j] < 67.5) or (-112.5 <= theta[i, j] < -67.5):
                N1, N2 = G[i+1, j+1], G[i-1, j-1]
            elif (67.5 <= theta[i, j] < 112.5) or (-67.5 <= theta[i, j] < -22.5):
                N1, N2 = G[i+1, j], G[i-1, j]
            else:
                N1, N2 = G[i-1, j+1], G[i+1, j-1]

            # Suppress non-maximum values
            if (G[i, j] < N1) or (G[i, j] < N2):
                suppressed_G[i, j] = 0
            else:
                suppressed_G[i, j] = G[i, j]

    return suppressed_G

def thresholding(G, t):
    '''Binarize G according threshold t'''
    G_binary = G.copy()
    G_binary[G_binary < t] = 0
    G_binary[G_binary > 0] = 255

    return G_binary

def hysteresis_thresholding(G: np.ndarray, low, high):
    '''Binarize G according threshold low and high'''
    # TODO: Please complete this function.
    # your code here
    h, w = G.shape
    STRONG = 255
    WEAK = 15
    
    G_low = thresholding(G, low)
    G_high = thresholding(G, high)
    G_hyst = G.copy()
    
    sx, sy = np.where(G >= high)
    wx, wy = np.where((G >= low) & (G < high))
    lx, ly = np.where((G < low))
    
    
    G_hyst[sx, sy] = STRONG
    G_hyst[wx, wy] = WEAK
    G_hyst[lx, ly] = 0
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            if G_hyst[i,j] == WEAK:
                if(G_hyst[i-1,j-1] > high) or (G_hyst[i-1,j] > high) or \
                    (G_hyst[i-1,j+1] > high) or (G_hyst[i,j-1] > high) or \
                        (G_hyst[i,j+1] > high) or  (G_hyst[i+1,j-1] > high) or \
                            (G_hyst[i+1,j] > high) or (G_hyst[i+1,j+1] > high):
                                G_hyst[i,j] = STRONG
    
    

    return G_low, G_high, G_hyst

def hough(G_hyst):
    '''Return Hough transform of G'''
    # TODO: Please complete this function.
    # your code here
    num_thetas = 1000
    
    H, W = G_hyst.shape
    D = int(np.sqrt(np.square(H) + np.square(W)))
    thetas = np.linspace(0, np.pi, num=num_thetas)
    rhos = np.arange(-D, D+1)
    
    y_idxs, x_idxs = np.nonzero(G_hyst)
    
    accumulator = np.zeros( (len(rhos), len(thetas)), dtype=np.uint64)
    
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        
        for j in range(len(thetas)):
            rho = np.round(x * np.cos(thetas[j]) + y * np.sin(thetas[j]))
            rho = np.clip(rho, -D, D)
            accumulator[int(rho + D), j] += 1
            
    return accumulator, thetas, rhos


def find_top_n_lines(accumulator, thetas, rhos, n):
    # Find the indices of the n cells with the highest values in the accumulator
    top_n_indices = np.unravel_index(np.argsort(accumulator.ravel())[-n:], accumulator.shape)
    
    # Convert the indices to (rho, theta) pairs
    top_n_lines = list(zip(rhos[top_n_indices[0]], thetas[top_n_indices[1]]))
    
    return top_n_lines
    

if __name__ == '__main__':
    input_path = 'data/road.jpeg'
    img = read_img_as_array(input_path)
    #show_array_as_img(img)
    #TODO: finish assignment Part II: detect edges on 'img'

    # Step 1
    gray = rgb2gray(img)
    save_path = './data/2.1_gray.jpg'
    save_array_as_img(gray, save_path)
    
    # Step 2
    img = ndimage.gaussian_filter(gray, sigma=1.6)
    
    # Step 3
    G_save_path = './data/2.3_G.jpg'
    Gx_save_path = './data/2.3_G_x.jpg'
    Gy_save_path = './data/2.3_G_y.jpg'
    
    G, Gx, Gy = sobel(img)
    save_array_as_img(G, G_save_path)
    save_array_as_img(Gx, Gx_save_path)
    save_array_as_img(Gy, Gy_save_path)
    
    # Step 4
    supress_save_path = './data/2.4_supress.jpg'
    suppressed_G = nonmax_suppress(G, Gx,Gy)
    save_array_as_img(suppressed_G, supress_save_path)
    
    # Step 5
    edgemap_low_save_path = './data/2.5_edgemap_low.jpg'
    edgemap_high_save_path = './data/2.5_edgemap_high.jpg'
    edgemap_save_path = './data/2.5_edgemap.jpg'
    G_low, G_high, G_hyst = hysteresis_thresholding(suppressed_G, 20 , 40)
    save_array_as_img(G_low, edgemap_low_save_path)
    save_array_as_img(G_high, edgemap_high_save_path)
    save_array_as_img(G_hyst, edgemap_save_path)
    
    # Step 6
    accumulator, thetas, rhos = hough(G_hyst)
    save_array_as_img(accumulator, './data/2.6_hough.jpg')
    
    # Step 7
    top_n_lines = find_top_n_lines(accumulator, thetas, rhos, 5)
    orgin_image = Image.open(input_path)
    draw = ImageDraw.Draw(orgin_image)
    
    for rho, theta in top_n_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        draw.line((x1, y1, x2, y2), fill=128, width=3)
    orgin_image.save('./data/2.7_detection_result.jpg')