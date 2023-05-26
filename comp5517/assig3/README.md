# COMP5517 25% Assignment 3 (1 individual assignment)
- find some key/state-of-the-art applications/papers on the topic of your group project
- write a 3-Page Survey Report for the applications/papers that you investigate

## Title: Survey on Deep Learning for Hand Gesture Recognition
### Abstract
Hand gesture recognition has emerged as a critical technique of Human–Computer Interaction (HCI) research because its potential applications in various fields, including human-computer interaction, sign language recognition, and other more. The Deep learning-based approach has a significant role in advancing the accuracy and robustness of hand gesture recognition systems. In this survey report, summarize and organize recent research results in deep learning for hand gesture recognition, and identify future research directions and unresolved problems.

### Introduction
Hand gestures are a form of body language that can be conveyed through the position and shape of the hand. Hand gestures is able to classified as static or dynamic. (Kumar, Vadakkepat, & Loh, 2010) It was initially achieved through wearable sensors attached to gloves, which detected hand movements and finger bending. However, these sensors have limitations, including discomfort and wire connection problems. However, camera vision-based technologies have been developed as a promising and cost-effective alternative, using different computer vision methods and algorithms to detect hand features, including the skeleton, skin colour, deep learning detection, and more. (Oudah, Al-Naji & Chahl, 2020) In particular, the deep learning approach has emerged as a powerful technique for hand gesture recognition due to its ability to learn complex features and handle large datasets. This survey aims to provide an overview of the current state-of-the-art techniques in deep learning for hand gesture recognition.

### Survey Details
In this section, we will review recent research results in deep learning for hand gesture recognition, which is divided into Hand gesture dataset, Feature extraction and classification methods.

#### Hand Gesture Datasets
Researchers use different publicly available hand gesture datasets to build and evaluate deep learning models for hand gesture recognition. Some of the most popular datasets include:

**American Sign Language (ASL) Finger Spelling Dataset**, the Collection of RGB and depth images for each letter in the alphabet, can be used for American Sign Language fingerspelling recognition. The dataset includes two subsets: Dataset A, which contains 24 static signs from 5 users, and Dataset B, which is captured from 9 users in two different environments and lighting conditions using depth data only. The datasets can be used for developing and testing machine learning models for fingerspelling recognition. (Pugeault & Bowden, 2011)

**NVIDIA Dynamic Hand Gesture Dataset** consists of 25 hand gesture types containing 1532 dynamic hand gestures captured indoors in a car simulator with both bright and dim artificial lighting. A total of 20 subjects participated in data collection. Each subject performed the gestures with their right hand while observing the simulator's display and controlling the steering wheel with their left hand. Includes a variety of gestures such as moving the hand or fingers in different directions, beckoning, opening, showing "thumb up" or "OK", and so on. (Molchanov, Yang, Gupta, Kim, Tyree & Kautz, 2016)

**EgoGesture** is the largest existing egocentric gesture dataset and is designed to enable the development of more accurate and robust gesture recognition algorithms for wearable devices.
It is a collection of video frames captured using the egocentric camera. The dataset contains 83 hand gestures designed to cover various manipulative and communicative operations for wearable devices. The gestures are divided into two categories: manipulative and communicative and are performed by 50 subjects in scenes of four indoor and two outdoor, including stationary or walking subjects with a static or dynamic clutter background.

### Feature Extraction
Before the widespread adoption of deep learning, it combined thresholding, skin colour detection, edge detection, and motion for detecting hand gestures. Thresholding is used to extract the moving object region, skin colour detection is used to identify skin regions, and edge detection is used to separate the arm region from the hand region. The combination of these features will be to allocate the hand region and identify hand gestures. (Chen, Fu & Huang, 2003). In addition, Scale-Invariant Feature Transform (SIFT) will also be applied at Feature Extraction, which can be used to extract features from an image of a hand that are unique to the particular hand gesture performed. It can achieve high accuracy even when the hand is rotated or translated into the image. In hand gesture recognition, SIFT can extract features from an image of a hand that are unique to the particular hand gesture being performed. It can achieve high accuracy even when the hand is rotated or translated into the image. However, this method contains some limitations, such as Sensitivity to image quality, Computational intensity, Variability in hand appearance and more.

With the advent of deep learning recently, the 2D or 3D Convolutional layers are widely used in hand gesture recognition. The input image is convolved with a set of learnable filters or kernels that scan the image and produce a set of feature maps. Each filter in the convolutional layer is responsible for learning a specific feature or pattern in the input image, such as edges, corners, or blobs. In addition, combined with the Max Pooling layer be useful for reducing the computational complexity of the network and providing some level of translation invariance. For example, if the hand appears in different locations in different images, max pooling can help the network recognize it regardless of its position.

Moreover, they allow for hierarchical feature learning. The output of one convolutional layer can be input to another, allowing the network to learn increasingly complex and abstract features. The MediaPipe Hands is a Hand Tracking framework proposed by Google which employs a single-shot detector (SSD) model present by Liu et al. (2016) for Hand/Palm detection. (Zhang et al., 2020) It applies the Convolutional and Max Pooling layers.

### Deep Learning of Classification Method
*Deep learning *is a relatively new machine learning approach involving neural networks with multiple hidden layers, which is achieved great success in computer vision, natural language processing and other tasks. The architectures and learning algorithms of the deep learning network are inspired biologically. (Kruger et al., 2012) it is trained layer-by-layer and relies on the more hierarchical feature learning generally. After feature extraction, several methods have been proposed for classified categories, including:

**A convolutional neural network (CNN)** consists of multiple layers, mainly including convolutional, pooling, and fully connect layer. In a convolutional layer, the network performs a mathematical operation called convolution, which involves sliding a set of filters over an input image to extract features. The pooling layer then down-samples the output of the convolutional layer to reduce the spatial dimension of the feature map. Finally, fully connected layers perform classification or regression tasks on the extracted features. It is effective in image analysis tasks because it can automatically learn and extract relevant features from raw image data without manual feature engineering.

**Autoencoders** are unsupervised learning for learning compression and reconstruction data. It means it did not need the class label for learning. The network consists of an encoder that maps the input data to the low -dimensional representation (potential space) and the decoder of the low-dimensional, indicating the mapping back to the original input data. The goal of the network is the difference between minimizing input data and reconstruction data. In training Autoencoders on large hand image data sets, the network can learn to extract the most related hand recognition features. The compression of the automatic encoder learning can be used as a classifier input to perform the recognition task. Oyedotun & Khashman (2017) utilize Stacked denoising autoencoder variants to recognize American Sign Language (ASL).

**Single Shot Multibox Detector (SSD)** (Zhang et al., 2020) and **You Only Look Once (YOLO)** (Redmon et al., 2016) are popular real-time object detection algorithms that use a convolutional neural network to predict the bounding boxes and class labels of objects in an image. The SSD works by dividing the input image into multiple fixed-size grids and predicting each grid's multiple bounding boxes and class probabilities. The network is trained using labelled images to minimize the localization error and classification loss. However, using a single network, YOLO will divide the image like a grid and predict target of class of bounding boxes probabilities. This approach is faster and more accurate than traditional object detection methods.

Recent research in deep learning for hand gesture recognition has focused on improving accuracy, reducing computational cost, and making the recognition process more robust to changes in environmental conditions. Another trend is transfer learning, it will use the pre- trained model to fine-tune gesture recognition tasks such as Resnet and AlexNet. Besides, is is possible to use the 3D CNNs to process temporal information and improve recognition accuracy. (Shen, Zheng, Feng, & Hu, 2022) Moreover, developing wearable devices with embedded deep-learning models for real-time gesture recognition is an emerging trend. (Tan et al., 2022)

### Unresolved Problems/Difficulties and Future Work
Despite the progress in deep learning-based gesture recognition technology, there still has some unsolved problems and difficulties.

One of the challenges is the lack of robustness and adaptability of gesture recognition systems to different environments, such as lighting, occlusions and noises. Another issue is acquiring such datasets is a time-consuming and challenging task. Furthermore, real-time performance is another major challenge that requires further attention, especially in real-time and interactive applications. Besides, future research on deep learning-based gesture recognition should focus on developing more user-friendly and intuitive interaction techniques that require less training and can be accessed by a wider range of users, such as children, the elderly and people with disabilities. After that, it is possible to investigate multimodal techniques combining gesture recognition with other modalities, such as speech and gaze, to enhance the understanding of user intent and commands and reduce the system's error rate. Furthermore, developing hand gesture recognition systems for human-robot interaction can be a promising research direction enabling robots to understand and respond to human gestures like a human.


### Conclusion
In conclusion, hand gesture recognition has become a critical technique in Human-Computer Interaction (HCI), enabling various applications, including sign language recognition and more. Deep learning has played a significant role in advancing the accuracy and robustness of hand gesture recognition systems. In this survey report, we have reviewed recent research results in deep learning for hand gesture recognition, divided into hand gesture datasets, feature extraction, and classification methods. The review indicates that deep learning has led to the development of more advanced techniques for hand gesture recognition. However, some limitations remain to overcome, such as sensitivity to image quality, computational intensity, and variability in hand appearance. Further research should focus on addressing these limitations and improving the accuracy of hand gesture recognition systems.

### Reference
Chen, F. S., Fu, C. M., & Huang, C. L. (2003). Hand gesture recognition using a real-time tracking method and hidden Markov models. Image and vision computing, 21(8), 745- 758.

Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C. Y., & Berg, A. C. (2016). Ssd: Single shot multibox detector. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14 (pp. 21-37). Springer International Publishing.

Kim, T. K., & Cipolla, R. (2008). Canonical correlation analysis of video volume tensors for action categorization and detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 31(8), 1415-1428.

Kruger, N., Janssen, P., Kalkan, S., Lappe, M., Leonardis, A., Piater, J., ... & Wiskott, L. (2012). Deep hierarchies in the primate visual cortex: What can we learn for computer vision?. IEEE transactions on pattern analysis and machine intelligence, 35(8), 1847- 1871.

Kumar, P. P., Vadakkepat, P., & Loh, A. P. (2010). Hand posture and face recognition using a fuzzy-rough approach. International Journal of Humanoid Robotics, 7(03), 331- 356.

Molchanov, P., Yang, X., Gupta, S., Kim, K., Tyree, S., & Kautz, J. (2016). Online detection and classification of dynamic hand gestures with recurrent 3d convolutional neural network. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4207-4215).

Oudah, M., Al-Naji, A., & Chahl, J. (2020). Hand gesture recognition based on computer vision: a review of techniques. journal of Imaging, 6(8), 73.

Oyedotun, O. K., & Khashman, A. (2017). Deep learning in vision-based static hand gesture recognition. Neural Computing and Applications, 28(12), 3941-3951.

Pugeault, N., & Bowden, R. (2011, November). Spelling it out: Real-time ASL fingerspelling recognition. In 2011 IEEE International conference on computer vision workshops (ICCV workshops) (pp. 1114-1119). IEEE.

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).

Shen, X., Zheng, H., Feng, X., & Hu, J. (2022). ML-HGR-Net: A meta-learning network for FMCW radar based hand gesture recognition. IEEE Sensors Journal, 22(11), 10808- 10817.

Tan, P., Han, X., Zou, Y., Qu, X., Xue, J., Li, T., ... & Wang, Z. L. (2022). Self-Powered Gesture Recognition Wristband Enabled by Machine Learning for Full Keyboard and Multicommand Input. Advanced Materials, 34(21), 2200793.

Wang, C. C., & Wang, K. C. (2008). Hand posture recognition using adaboost with sift for human robot interaction. In Recent Progress in Robotics: Viable Robotic Service to Human: An Edition of the Selected Papers from the 13th International Conference on Advanced Robotics (pp. 317-329). Springer Berlin Heidelberg.

Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C. L., & Grundmann, M. (2020). Mediapipe hands: On-device real-time hand tracking. arXiv preprint arXiv:2006.10214.

Zhang, Y., Cao, C., Cheng, J., & Lu, H. (2018). Egogesture: a new dataset and benchmark for egocentric hand gesture recognition. IEEE Transactions on Multimedia, 20(5), 1038- 1050.


