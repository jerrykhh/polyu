# Discussion

## Step 2
`sigma` as a standard deviation of the Gaussian filter, affects the final edge detection results in several ways.

The Gaussian filter becomes more blurrier when the `sigma` increases, which leads to a greater reduction in high-frequency noise in the image. However, if the `sigma` as increases continuously the filter also becomes less sensitive to small-scale edges and details, leading to possible loss of edge information. On the other hand, reducing sigma can help preserve small features and details in the image, but it may also increase noise and make the image look more grainy.

Therefore, we need to choose a proper sigma requires balancing different effects. Larger sigma is appropriate when the noise level in the image is high but may not be optimal for images with fine details edges. Conversely, a smaller sigma may be better for images with fine details or smaller-scale edges, but may not be sufficient to remove high-frequency noise. 

## Step 5

A high threshold is used to find strong edges and a low threshold is used to find weak edges. Any edge with an intensity value greater than a high threshold is considered a strong edge, while any edge with an intensity value less than a low threshold is considered a non-edge. Weak edges are those edges with intensity values between the high and low thresholds. We then apply a connectivity check to determine if any weak edges are part of strong edges. A weak edge is considered an edge if it is connected to a strong edge; otherwise, it is considered a non-edge.

In the source code, `def hysteresis_thresholding(G: np.ndarray, low=20, high=40)`.