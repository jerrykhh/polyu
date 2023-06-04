# Discussion

## 8.1 Handwritten digit recognition (18 points)
### Assume we set batch_size=8, what shape do above 7 layers’ inputs and outputs have?

| Layer | Input Shape | Output Shape |
| -- | -- | -- |
| 1. Conv layer | [8,1,28,28] | [8,16,24,24] |
| ReLu | [8,16,24,24] | [8,16,24,24] |
| MaxPool2D | [8,16,24,24] | [8,16,12,12] |
| 2. Conv layer | [8,16,12,12] | [8,32,10,10] |
| ReLu | [8,32,10,10] | [8,32,10,10] |
| MaxPool2D | [8,32,10,10] | [8,32,5,5] |
| 3. Conv layer | [8,32,5,5] | [8,32,3,3] |
| ReLU | [8,32,3,3] | [8,32,3,3] |
| Flattern | [8,32,3,3] | [8,288] |
| 1. FC layer | [8,288] | [8,64] |
| ReLU | [8,64] | [8,64] |
| 2. FC layer | [8,64] | [8,10] |


### How many trainable parameters each layer contains?
1. Conv layer 1: $(5\times 5\times 1)\times 16+16=416$
2. Conv layer 2: $3\times 3\times 16\times 32+32=4640$ 
3. Conv layer 3: $3\times 3\times 32\times 32+32=9248$ 
4. Fully Connected Layer 1: $32\times 3\times 3 \times 64 + 64 =18496$
5. Fully Connected Layer 2: $64\times 10 + 10 = 650$

```math
416+4640+9248+18496+650=33450
```

$\therefore$ Total Trainable params: $33450$

(torchsummary)
![Image](../image/model_summay.png)

## Screenshot

**imshow**
![Image](../image/model_imshow.png)

![Image](../image/model_acc.png)

## 8.2 Bonus: Fashion-MNIST (3 points)
### Try data augmentation functions like torchvision.transforms.RandomHorizontalFlip and torchvision.transforms.RandomRotation, discuss the influence of the data augmentation parameters to the final accuracy. 

`transforms.RandomHorizontalFlip()`: This transformation randomly flips the input image horizontally with a given probability (default is 0.5). This augmentation technique helps the model generalize better by introducing variations in the training data. It makes the model more robust to slight changes in the orientation of the objects in the images. However, if the probability of flipping is too high, the model may be trained on too many flipped images, reducing its ability to recognize the original orientation.

`transforms.RandomRotation(10)`: This transformation randomly rotates the input image by a random angle sampled from a range of angles specified in degrees. In this case, the range is set to (-10, 10). It makes the model more robust to different rotations of the objects in the images. However, if the range of rotation angles is too extensive, the model might be trained on images with unrealistic rotations, which could negatively impact its performance on the test set. 

Data augmentation techniques can improve the model's generalization ability and robustness by introducing variations in the training data. However, the choice of parameters for these techniques, such as the flipping probability and rotation angle range, can influence the final accuracy of the model.

### Screenshot

![Image](../image/model2_train_loss.png)

![Image](../image/model2_acc.png)