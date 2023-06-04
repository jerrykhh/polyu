# COMP5523 Written Assignment

## Question 1
(15 pts) Describe the effects of lens, aperture size, and focal length in Camera Model.

## Question 2
(5 pts) Applying a filter of size $m\times m$ on an image of size $n\times n (n >> m)$, what is the size of the output? Discuss different cases as in page 72 of lecture note 3.   

## Question 3
(10 pts) Calculate the convolution of the two filters below (the result is a $3 \times 3$ matrix ) Explain the meaning of the convolution and the result.`
```math
\begin{pmatrix}
1 &2  &1 \\ 
2 &4  &2 \\ 
1 &2  &1 
\end{pmatrix} \times \begin{pmatrix}
1 &0  &-1 
\end{pmatrix}
```

## Question 4

(10 pts) The 1D filter $[1, 0, -1]$ gives an estimation of the first-order derivative. What is the corresponding second-order derivative filter? (Hint: it is a $1\times 5$ vector). Explain your answer and show the calculation steps if necessary. (Note: a filter needs to be flipped for calculating convolution.)


## Question 5
(10 pts) Describe the technique used in Canny edge detector to reduce redundant edges. Compare it with a similar technique used in Harris corner detector and describe their differences.

## Question 6
(10 pts) Compare the $k$-means and mean shift algorithms for clustering and describe their advantages and disadvantages. 

## Question 7
(10 pts) In a convolutional layer of a neural network, suppose that the input feature map is of size $n \times n \times k$ and we use $c$ of size $m \times m \times k$ ($m$ is usually an odd number) to convolve with the image, what is the size of the outputting feature map ($n > m $ & no padding) ?

## Question 8
(10 pts) Describe as many as possible regularization methods that can be used to train neural networks.

## Question 9
(10 pts) When training a neural network, what is the impact of learning rate?

##　Question 10	
(10 pts) Explain why CNN is better than MLP for image classification. Is back propagation needed in a convolutional layer? Explain your answer. 

## Answer

### Question 1
In a camera model, the lens, aperture size, and focal length are crucial components that capture images. They have a significant impact on various aspects of the resulting photograph.

The lens is an optical device that focuses light onto the camera's sensor or film. Lenses come in various types and designs, such as comprehensive, zoom, and so forth. The lens affects several image-capturing aspects, including Sharpness, Anamorphic, Chromatic aberration and Vignetting. Generally, Higher quality lenses contain different technic to reduce these aspects and provide sharper images with better resolution and contrast. In Chromatic aberration, it means the colour fringing occurs at the edges of high-contrast areas due to the inability of the lens to focus all the colours at the same point, causing the light to split into its component colours. Besides, Vignetting occurs when the lens's edges block some of the light from reaching the sensor, causing the image's corners to be darker than the centre.

Besides, about the aperture size, it refers to the adjustable opening in the lens that controls the amount of light entering the camera. The size of the aperture is measured in f-stops (e.g., f/0.95, f/1.2), with lower f-stop numbers representing larger apertures and vice versa. In Exposure, A larger aperture (lower f-stop) lets in more light, resulting in a brighter image, and creates a shallow depth of field, meaning only a tiny portion of the image is in focus. At the same time, the background and foreground are blurred. However, a smaller aperture (higher f-stop) lets in less light, producing a darker image, and A smaller aperture increases the depth of field, resulting in more of the image is in focus.

Shorter focal lengths (e.g., 18mm) provide a wider field of view, capturing more of the scene in a single frame, and contain more considerable Anamorphic lead to exaggerate the distance between objects, making the foreground appear larger more prominent. On the other hand, longer focal lengths (e.g., 200mm) have a narrower field of view due to compression of the distance between objects, creating a more flattened appearance. At the same time, longer focal lengths generally produce a shallower depth of field, while shorter focal lengths offer a deeper depth of field.

### Question 2
When shape = ‘full’, the output size is (n+m-1)×(n+m-1).
When shape = ‘same’, the output size is n×n.
When shape = ‘valid’, the output size is (n-m+1)×(n-m+1).

#### Question 3
This filter calculates the first order derivative of input images, which is a Gaussian kernel. The output is the derivative of Gaussian. And the 3x3 matrix is:

```
2 0 -2
4 0 -4
2 0 -2
```

### Question 4
Second-order derivative is the derivative of derivative, which is equivalent to applying first- order derivative twice. So, the corresponding second-order derivative filter is [1 0 −2 0 1].


### Question 5
Canny Edge Detector and Harris Corner Detector are popular feature detection techniques in computer vision, but they serve different purposes. The Canny Edge Detector is designed to detect edges in an image, while the Harris Corner Detector is designed to identify corner points. In addition, both methods employ techniques to reduce noise and redundancy in their respective outputs.

The Canny Edge Detector uses a multi-stage process to detect edges, and one of its key steps is non-maximum suppression (NMS) to reduce redundant edges. Firstly, a Gaussian filter is applied to the image to smooth it and reduce noise. It will utilize the Sobel operator to compute the gradient magnitude and direction for each pixel in the smoothed image. The Non-Maximum Suppression will reduce redundant edges by preserving only local maxima in the gradient magnitude image. For each pixel, the algorithm checks whether it is a local maximum along the gradient direction. If not, the pixel's value is set to zero, effectively thinning the edges. After that, high and low threshold values are chosen to classify strong and weak edges. Pixels with gradient magnitudes above the high threshold are considered strong edges, while those between the low and high thresholds are considered weak edges. Finally, it involves linking strong edges with weak edges. If a weak edge is connected to a strong edge, it is considered a true edge; otherwise, it is discarded

Besides, regarding the Harris Corner Detector, it corners by calculating the second- moment matrix of the image, which captures the local intensity variations. Firstly, Compute the image gradients x and y using a derivative kernel, such as the Sobel operator. For each pixel, calculate the structure tensor by forming a weighted sum of the outer product of gradients. Also, calculate the corner response function based on the determinant and trace of the structure tensor. Finally, it will apply NMS to the corner response image to find local maxima. This step reduces redundant corners by keeping only those with the highest R-value in their neighbourhood.

While both the Canny Edge Detector and the Harris Corner Detector use Non- Maximum Suppression (NMS) to reduce redundancy, the main differences are their goals and the nature of the features they detect. The Canny Edge Detector aims to identify edges, while the Harris Corner Detector focuses on detecting corners. NMS in the Canny algorithm is applied to thin edges, while NMS in the Harris algorithm is applied to find strong corner candidates. The Canny algorithm also uses double thresholding and edge tracking by hysteresis to refine edge detection results, while the Harris algorithm relies on the corner response function to identify corners.

### Question 6
K-means is a centroid-based clustering algorithm that aims to partition n data points into k clusters by minimizing the within-cluster sum of squares. The algorithm will iteratively assign each data point to the nearest centroid and updates the centroids based on the mean of the data points in each cluster.

Advantages:
1. Simple and easy to implement
2. Scalable to large datasets
3. Converges relatively quickly
4. Works well with spherical and equally sized cluster

Disadvantages
1. Requires the user to specify the number (k) as a cluster beforehand, which may  only sometimes be known or easy to estimate.
2. Sensitive to the initial placement of centroids. The result in different clustering outcomes. It may need to multiple runs with different initializations are often  performed.
3. Can get stuck in local minima, leading to suboptimal clustering results
4. Assumes equal-sized clusters with similar densities, which may only sometimes be the case in real-world datasets

Mean Shift is a non-parametric, density-based clustering algorithm that identifies clusters by estimating the modes of the underlying probability density function. The algorithm starts with a set of initial points (called kernel centres). It iteratively shifts them towards areas with higher point densities by computing the mean of the points within a given bandwidth.

Advantages:
1. It does not require the user to specify the number of clusters, as it automatically  estimates the optimal number based on the data.
2. Suitable for noise and outliers, as it is a density-based method.
3. No assumption about the shape or size of the clusters. It can identify clusters 

Disadvantages:
1. Computationally more expensive than K-means, especially for large datasets and high-dimensional data.
2. The user must set the bandwidth parameter, which can significantly affect the clustering results. A small bandwidth may lead to over-segmentation, while a large bandwidth may merge distinct clusters.
3. Convergence can be slow, particularly for large datasets.
4. It is not guaranteed to converge to a global solution.

In summary, K-means is a simple, fast, and scalable algorithm for spherical and equal- sized clustering, but it requires specifying the k number of clusters and is sensitive to initialization. On the other hand, Mean Shift does not require specifying the number of clusters and can handle different cluster shapes and sizes. However, it is computationally more expensive and requires proper bandwidth parameters to be set.

### Question 7
$(n – m + 1) \times (n – m + 1) \times c$

### Question 8

1. **L1 Regularization**, this regularization method promotes sparsity in the learned model, as it encourages some weights to become exactly zero.
2. **L2 Regularization**, unlike L1 regularization, L2 regularization does not encourage sparsity but instead penalizes large weight values, promoting 
smoother and smaller weights.
3. **Dropout**, dropout randomly a fraction of the neurons' activations in a given layer at each iteration
4. **Early Stopping**, involves monitoring the model's performance on a validation dataset during training and stopping the training process when the validation 
performance starts to degrade.
5. **Weight Decay**, is a technique that gradually reduces the magnitude of the model's weights during training. It can be seen as a variant of L2 regularization.
6. **Data Augmentation**, involves generating new training samples by applying various transformations to increases the diversity of the training data
7. **Noise Injection**, a technique used to limit the magnitude of the gradients during the backpropagation process
8.** Weight Tying**, technique where certain weights in the neural network are forced to have the same values during training. This reduces the number of trainable parameters and acts as a form of regularization

### Question 9
The learning rate is a crucial hyperparameter in the training process of a neural network. It controls the step size or magnitude of the weight updates during optimization. In gradient-based optimization algorithms, the learning rate determines how much the model's weights are adjusted in response to the computed gradient of the loss function concerning those weights. Furthermore, it will affect the convergence speed, accuracy and stability, and overfitting and generalization of a neural network during training. A higher learning rate can result in faster convergence but may cause instability and less accurate solutions. A lower learning rate can result in slower convergence but may help prevent overfitting and improve accuracy. Choosing an appropriate learning rate to balance these factors for optimal performance is important.

### Question 10
For several reasons, CNN is better than MLP (Multilayer Perceptron) for image classification tasks. Firstly, it utilizes local connectivity to learn more complex and robust representations of local image patterns. In contrast, MLPs use fully connected layers. Each neuron is connected to all neurons in the preceding layer, making it harder to capture local features. Besides, the same filter is used across the entire input image, and the same set of weights is shared. This greatly reduces the number of parameters in the network and makes it computationally more efficient and less prone to overfitting than MLPs.

Moreover, with the convolution operation and parameter sharing, CNNs can recognize the same features in different parts of an image, making them invariant to translations. It is desirable for image classification tasks, as objects can appear in various positions within an image. MLPs, on the other hand, lack this invariance. Furthermore, multiple layers learn a hierarchy of features. Lower layers capture simple features like edges and corners, while deeper layers learn more complex, high-level features like object parts or entire objects. This hierarchical representation enables CNNs to learn more expressive and powerful features for image classification. In contrast, MLPs have a different hierarchical structure which can make it easier for them to learn complex features.

On the other hand, backpropagation is the algorithm used to train neural networks, including MLPs and CNNs. It works by the gradient of the loss function concerning each weight and bias by applying the chain rule and computing gradients layer by layer come from the output layer to the input layer. In the case of a CNN, backpropagation is used to update the weights and biases of the convolutional layers and any fully connected layers present in the network.