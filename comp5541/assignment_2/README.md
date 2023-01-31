# COMP5541 Assignment 2
## 1 Gradient Descent (45 marks)

In this question, we will discuss properties of gradient descent. In the following, we will
consider the family of loss functions $l_a(w)=aw^2$ for $a>0$.

(1) Let $a=2$ and consider the two step sizes $\eta_1=1/2$ and $\eta_2=2$. What do you observe when starting gradient descent at $w_0=1$? (15 marks)

(2) Define the set of all step sizes $\eta > 0$ for which gradient descent will fail to converge to the minumum $w^*$ of $l_a(w)$ when starting at $w_0\neq 0$. (15 marks).

(3) Let $\epsilon > 0$, and suppose $\eta$ is chosen such that gradient descent coverges to the minimum $w^*$ of $l_a(w)$ starting at $w_0$. After how many steps $i$ do we have $|w_i - w^*| < \epsilon$? (15 marks)

## 2 CNN Network (55 marks)
We'll build a convolutional neural network to classify handwritten digits. You need to down-load the MNIST dataset via Pytorch (Other tools are also fine).

(1) You should now build a network with the following layers (10 marks):

-  The inputs are provided as vectors of size 784, you should reshape them as 28x28x1 images (1 because these are grey scale);
-  Add a convolutional layer with 25  lters of size 12x12x1 and the ReLU non-linearity. Use a stride of 2 in both directions and ensure that there is no padding;
-  Add a second convolutional layer with 64  lters of size 5x5x25 that maintains the same width and height. Use stride of 1 in both directions and add padding as necessary and use the ReLU non-linearity.
-  Add a max pooling layer with pool size 2x2;
-  Add a fully connected layer with 1024 units. Each unit in the max_pool should be connected to these 1024 units. Add the ReLU non-linearity to these units;
-  Add another fully connected layer to get 10 output units. Don't add any non-linearity to this layer, we'll implement the softmax non-linearity as part of the loss function.

(2) Loss Function, Accuracy and Training Algorithm (10 marks)

- We'll use the cross entropy loss function;
- Accuracy is simply de ned as the fraction of data correctly classified;
- For training you should use the AdamOptimizer (read the documentation) and set the learning rate to be 1e-4.

(3) Training (10 marks)

You should now train your neural network using minibatches of size 50. Try about 1000- 5000 iterations. You may want to start out with fewer iterations to make sure your code is making good progress. Once you are sure your code is correct, you can let it run for more iterations and read up for the rest of the assignment while the code runs. Keep track of the validation accuracy every 100 iterations or so; however, do not touch the test dataset. Once you are sure your optimisation is working properly, you should run the resulting model on the test data and report the test error. (You should get about 98% accuracy.)

(4) Visualising Filters (10 marks)

Please visualise the  lters in the  rst convolutional layer. There are 25  lters each of size 12x12x1. Thus, the  lters themselves can be viewed as 12x12 greyscale images. The size of the images in the dataset is relatively small, so don't expect these  lters to have some obvious patterns.

(5) Submitting Code and Responses (15 marks)

You need to submit the executable python code, together with responses to all above sections. Please attach screenshots of particular code scripts, running results, and all visual-ization results.

## Solution

## Question 1

### (1). 

$l_a(w)=aw^2$

$\frac{d l_a(w)}{dw} = \frac{d(aw^2)}{dw}=2aw$

For $\eta_1=1/2,w_0=1,a=2$

$w_1=1-1/2\times 2\times 2\times 1=-1$

$w_2=-1-1/2\times 2\times 2\times -1=1$

$w_3=1-1/2\times 2\times 2\times 1=-1$

We can observe that for every update of w, it becomes the negative of its previous value. From 1 to -1, -1 to 1, and this update pattern will keep forever if we don't stop. So it can only has 2 values and will not be able to converge.

For $\eta_2=2,w_0=1,a=2$

$w_1=1-2\times 2\times 2\times 1=-7$

$w_2=-7-2\times 2\times @\times -7=49$

$w_3=49-2\times 2\times 2\times 49=-343$

We can observe that for every update of w, the absolute value of w will keep becoming bigger. So w will never converge with $\eta_2=2$

### (2).

$w_1=w_0-2\eta a, w_0=(1-2\eta a) w_0$

$w_2=(1-2\eta a)^2\times w_0$

$\therefore w_i = (1-2\eta a)^i \times w_0$

w will fail to converge if either of following condition is met:
1. $1-2\eta a >=1$
2. $1-2\eta a <=-1$

From 1, we can know: $-2\eta a >=0 $

$\eta <= 0 $ (rejected)

$\because$ we want $\eta > 0$

From 2, we can know $-2\eta a <=-2$

$\eta >=1/a$

$\therefore$ it will fail to converge to min when $\eta >=\frac{1}{a}$

### (3).

when $w_0$ is $0$ for $a>0$

$\therefore w* = 0$

$|w_i-w^*|<\varepsilon$

$|(1-2\eta a)^i \times w_0-0|< \varepsilon$

$|(1-2\eta a)^i| \times |w_0|< \varepsilon$

$|(1-2\eta a)^i| \times < \frac{\varepsilon}{|w_0|}$

$\ln (|(1-2\eta a)^i|) < \ln (\frac{\varepsilon}{|w_0|})$

$i \times \ln (|(1-2\eta a)|) < \ln (\frac{\varepsilon}{|w_0|})$

$i > \frac{ \ln (\frac{\varepsilon}{|w_0|})}{ \ln (|(1-2\eta a)|)}$

### Question 2

Please reference the `COMP5541_assig2.ipynb`