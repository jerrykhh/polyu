# Topic: Review of Image generation

## Abstract
This research summary provides an overview of state-of-the-art image generation techniques focusing on advancements brought forth by deep learning algorithms. The paper covers various categories of image generation, discusses recent advancements in each category and presents various deep-learning approaches to image generation, including Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and Diffusion models. However, describe several challenges and limitations.

## Introduction
The rapid advancement of deep learning techniques has led to significant progress in various fields, including image generation. Image generation is creating novel images through algorithms capable of synthesizing and manipulating visual data. This technology has numerous applications across different industries, such as artistic expression, industrial design, virtual reality, and data augmentation. Despite its potential, generating realistic, diverse, and controllable images with computational efficiency remains challenging.

Recent advances in deep learning algorithms, such as Generative Adversarial Networks (GANs) and diffusion models, have shown remarkable improvements in generating high-quality images with greater control and efficiency than traditional image generation methods. The primary objective of this research summary is to provide a comprehensive overview of state-of-the-art image generation techniques, focusing on advancements brought forth by deep learning algorithms. The various categories of image generation, such as image-to-image translation, sketch-to-image generation, text-to-image generation, layout-to-image generation, facial image generation, video generation, panoramic image generation, scene graph image generation, fewshot image generation, 3D image/object generation, and multi-view image generation, are discussed along with contributions in each category

Moreover, several deep-learning approaches to image generation have been extensively researched, including Variational Autoencoders (VAEs), Generative Adversarial Networks, and Diffusion models. Despite progress in image generation, several challenges and limitations exist, including high-quality image generation, model collapse, training instability, precise control over generated content, evaluation metrics, generalization, computational resources, and ethical and legal concerns.

## Related Work
### Categories of Image Generation
According to Elasri et al. (2022), image generation encompasses various approaches and
applications, which can be categorized into various types. The following provides a summary
of notable advancements in the field and their various applications.


**Image-to-Image Translation (I2I)** involves the conversion of an input image into a corresponding output image while preserving certain desired properties of the original input image (Pang, Lin, Qin & Chen, 2021). This process can be used for various applications, such as style transfer (Kim et al., 2017) and image inpainting (Pathak et al., 2016). Mao, Wang, Zheng & Huang proposed a GAN model that considers image semantics to generate the salient object. The model controls the hierarchical semantics of images by processing semantic information on label and spatial levels and adds constraints to label and spatial levels to retain image semantics and improve the quality of generated samples. Another work proposed by Tang et al. (2019) utilizes Cycle In Cycle GAN (C2GAN) for keypoint-guided image generation. The network consists of two types of generators, keypoint-oriented and imageoriented, connected in a unified network structure that explicitly forms three cycled subnetworks. The cycles aim to reconstruct the input domain and provide extra supervision signals for better image generation. Zhou & Lee (2020) use a GAN model to generate bird's eye images from front-view images for autonomous driving. The proposed method employs a generator and two discriminators, with one discriminator identifying whether the two views are associated. Unlike previous methods that use geometry-based transformation or intermediate views, this method uses a pixel-level network to synthesize the bird's eye view.

**Sketch-to-Image Generation**, the goal of this technique is to generate realistic images from simple sketches. Early methods for synthesizing images from sketches relied on image retrieval, such as PhotoSketcher (Eitz et al., 2011) and Sketch2Photo (Chen et al., 2009). Recently, Sketchy-GAN (Chen & Hays, 2018) and ContextualGAN (Lu et al., 2018) have been based on object-level sketches depicting single objects. Also, Gao et al. (2020) introduce EdgeGAN, which learns a joint embedding to transform images and the corresponding various-style edge maps into a shared latent space. Therefore, automatic image generation from freehand sketches can be used to specify the synthesis goal and control the image generation. Cheng et al. (2023) introduce a unified framework called DiSS that generates photorealistic images from sketches and coloured strokes based on diffusion models.

**Text-to-Image Generation**, utilizes natural language description as input and generates a corresponding realistic image as output. It involves complex algorithms such as generative adversarial networks (GANs) and attention mechanisms to learn from a large dataset of images and corresponding textual descriptions and generate high-quality images to match the input descriptions. According to Agnese, Herrera, Tao & Zhu (2020), traditional learning-based textto-image synthesis approaches have a major limitation: they cannot generate new image content. They can only change the characteristics of the given/training images. However, Ramesh et al. (2022) proposed a two-stage model for text-conditional image generation using CLIP, a contrastive model that learns robust image representations capturing both semantics and style. The first stage is a prior that generates a CLIP image embedding given a text caption. In contrast, the second stage is a decoder that generates an image conditioned on the image embedding. The authors show that explicitly generating image representations improve image diversity with minimal loss in photorealism and caption similarity.

Moreover, Saharia et al. (2022) present Imagen, a text-to-image diffusion model that combines
transformer language models with high-fidelity diffusion models. The key finding is that text
embeddings from large language models are remarkably effective for text-to-image synthesis.
Comparing Imagen with recent methods, humans prefer Imagen over other models in side-byside comparisons regarding sample quality and image-text alignment.

**Layout-to-Image Generation** is a technique in which photorealistic images are synthesized based on semantic layouts. According to Cheng, Liang, Shi, He, Xiao, & Li (2023), GANs will be used for layout-to-image generation, but they can struggle with complex layouts. Therefore, proposes a method called LayoutDiffuse. It adopts a foundational diffusion model pre-trained on large-scale image or text-image datasets using a neural adapter based on layout attention and task-aware prompts. The method is data-efficient, generates images with high perceptual quality and layout alignment, and needs fewer data than the ten generative models based on GANs, VQ-VAE, and diffusion models.

**Facial image Generation** generates realistic images of human faces that may blur, are low quality, etc. Therefore, researchers attempted to generate images from various inputs, such as sketch, thermal, low-resolution, and blurry images. For example, Xia, Yang, Xue & Wu (2021) propose TediGAN achieves accuracy for facial image generation, a framework for multimodal image generation and manipulation with textual descriptions. The method includes a StyleGAN inversion module, visual-linguistic similarity learning, and instance-level optimization. The proposed method can produce high-quality images with a resolution of up to 1024x1024 and inherently supports image synthesis with multimodal inputs. To facilitate text-guided multimodal synthesis, the Multimodal CelebA-HQ dataset is introduced, which consists of real face images and corresponding semantic segmentation maps, sketches, and textual descriptions.

**Video Generation** requiring producing a sequence of images rather than a single image. Pan et al. (2019) propose a new method of video generation conditioned on a single semantic label map, which allows a good balance between flexibility and quality compared to existing video generation approaches. The task is divided into two sub-problems, i.e., image generation followed by image-to-sequence generation, such that each stage can specialize on one problem. The authors use a conditional VAE for predicting optical flow as an intermediate step to generate a video sequence conditioned on the single initial frame. A semantic label map is integrated into the flow prediction module to achieve major improvements in the image-tovideo generation process. Besides, Ho et al. (2022) present Imagen Video, a text-conditional video generation system based on a cascade of video diffusion models. It generates highdefinition videos using a base video generation model and a sequence of interleaved spatial and temporal video super-resolution models. The system has a simple architecture and can generate high-quality videos with strong temporal consistency, deep language understanding, and high controllability.

**Panoramic image Generation** involves creating high-resolution, wide-angle images that capture a 360-degree view of a scene. The goal is to generate a single panoramic image by stitching together overlapping images of the same scene taken from different angles. Referring to Song et al. (2022), traditional stitching methods include handling parallax distortion through various techniques such as dual homography, affine fields, and content-preserving warping. Visually unpleasant distortion, such as distortion on thin objects. As a result, they proposed a weakly-supervised deep learning-based stitching model for creating panoramic images from multiple real-world fisheye images. The model consists of colour consistency corrections, warping, and blending and is trained using perceptual and SSIM losses. The proposed algorithm overcomes the challenge of obtaining pairs of input images with a narrow field of view and ground truth images with a wide field of view captured from real-world scenes.

**Scene graph image Generation**, a scene graph is a hierarchical structure that represents objects in a scene and their relationships with each other, such as their position, size, and properties. It will generate an image based on a textual representation of the objects and their relationships in a scene. Mittal, Agrawal, Agarwal, Mehta, & Marwah (2019) presents a method for generating images interactively based on a sequence of scene graphs, allowing for incrementally additive text descriptions. The proposed recurrent network architecture uses Graph Convolutional Networks (GCN) to handle variable-sized scene graphs and Generative Adversarial image translation networks to generate realistic multi-object images without intermediate supervision.

**Few-shot Image Generation** refers to generating high-quality images using only a few examples as input. Ojha et al. (2021) propose a new approach. A large source domain is used for pre-training and transfer learning of diverse information from source to target to reduce overfitting. The proposed approach preserves relative similarities and differences between instances in the source domain via a cross-domain distance consistency loss. It employs an anchor-based strategy to encourage different levels of realism over different regions in the latent space. The approach automatically discovers correspondences between related sources and target domains to generate diverse and realistic images.

**3D Image/Object Generation** involves creating a digital representation of a three-dimensional object or scene, which can be viewed and manipulated in 3D space. Deng et al. (2020) propose a novel approach similar to StyleGAN for 3D face image generation of virtual people with disentangled, precisely-controllable latent representations for identity, expression, pose, and illumination. The authors embed 3D priors into adversarial learning and train the network to imitate the image formation of an analytic 3D face deformation and rendering process. They introduce contrastive learning to promote disentanglement by comparing pairs of generated images to deal with the generation freedom induced by the domain gap between real and rendered faces. Zhu et al. (2018), authors proposed a new generative model named Visual Object Networks (VON) that synthesizes natural images of objects with a disentangled 3D representation. And uses an end-to-end adversarial learning framework to extract the depth map features and silhouette for jointly modelling 3D shapes and 2D images to generate realistic images. In addition, Chen, Cohen-Or, Chen, & Mitra (2020) proposes a novel approach for achieving whole stomach 3D reconstruction during gastric endoscopy without using indigo carmine (IC) blue dye. The proposed method generates virtual IC-sprayed (VIC) images through image-to-image translation trained on unpaired real no-IC and IC-sprayed images using CycleGAN and Neural Graphics Pipeline (NGP) for control image generation

**Multi-View Image Generation** generates images of objects or scenes from different viewpoints or angles. Zhao et al. (2018) propose a novel image generation model named VariGANs that generates multi-view images with a realistic-looking appearance from a single view input. It combines the strengths of variational inference and Generative Adversarial Networks (GANs) and generates the target image coarsely instead of a single pass. The coarse image generator produces the basic shape of the object with the target view, and the fine image generator fills the details into the coarse image and corrects the defects. Tian et al. (2018) propose a two-pathway framework, CR-GAN, to generate multi-view images from a singleview input. The authors argue that the single-pathway framework of the widely-used GAN may learn "incomplete" representations, resulting in poor generalization ability on "unseen" data. CR-GAN introduces an additional generation path that utilizes labelled and unlabelled data for self-supervised learning to create view-specific images from embeddings randomly sampled from the latent space. The two paths collaborate and compete in a parameter-sharing manner to yield considerably improved generalization ability.

The field of image generation has seen significant advancements in recent years, with generative adversarial networks (GANs) and diffusion models being common and popular topics. Image generation can be categorized into various types, including image-to-image translation (I2I), sketch-to-image generation, text-to-image generation, layout-to-image generation, facial image generation, video generation, panoramic image generation, scene graph image generation, few-shot image generation, 3D image/object generation, and multiview image generation. Some notable contributions in this field include using the diffusion model to generate the Video, utilizing CLIP Contrastive models for Text-Conditional, and combining transformer language models with high-fidelity diffusion models for text-to-image synthesis. Overall, the continuous innovation in GANs and diffusion models has made them hot topics in image generation research.

## Approaches
Several deep-learning approaches to image generation have been extensively researched. Some of the most prominent methods include.

**Variational Autoencoders (VAEs)**: Variational Autoencoders (VAEs) are a generative model used for image generation, among other applications. VAEs combine deep learning and probabilistic graphical modelling with learning a compact, continuous latent representation of the input data, which can then be used to generate new samples. The VAE architecture consists of an encoder and a decoder network implemented as neural networks. The encoder takes an input image and maps it to a latent representation, typically a Gaussian distribution with a mean and variance. It will learn to map the input data to the most probable latent representation. After that, to enable backpropagation through the stochastic sampling process, the VAE employs the reparameterization trick. It involves sampling an auxiliary noise variable and transforming it using the mean and variance learned by the encoder. It produces a sample from the latent representation while still allowing gradients to flow through the network. However, the decoder network takes a sample from the latent representation and reconstructs the input image. Given the compact latent representation, the decoder learns to generate images that are as close as possible to the original input data. During training, the VAE minimizes the combined loss (reconstruction loss and KL-divergence), leading to a model that can generate new images by sampling from the latent space and passing the samples through the decoder. (Kingma & Welling, 2019)

**Generative Adversarial Networks**: proposed by Ian Goodfellow in 2014 and have since become a popular and powerful approach to generating realistic images, consist of two neural networks, a generator and a discriminator, who are trained simultaneously in an adversarial manner. The generator produces synthetic images, while the discriminator evaluates the quality of those images, trying to distinguish them from real images. The generator improves its image generation capability by trying to fool the discriminator. The generator will take a random noise vector as input and produce a synthetic image. The generator's goal is to generate images that are indistinguishable from real images in order to fool the discriminator.

On the other hand, the discriminator network takes an image as input (either a real image from the training dataset or a synthetic image generated by the generator). It outputs a probability indicating whether the input image is real or fake. The discriminator's goal is to accurately distinguish between real and generated images. In training, the generator and discriminator are trained simultaneously in an adversarial manner. The generator tries to produce more realistic images to fool the discriminator, while the discriminator tries to better distinguish real images from fake ones. This process continues until an equilibrium is reached, where the generator produces high-quality, realistic images, and the discriminator can no longer reliably distinguish between real and generated images. Since their introduction, numerous GAN variants and improvements have been proposed, such as DCGAN, WGAN, CycleGAN, and StyleGAN.

**Diffusion models**: introduced by Sohl-Dickstein et al. in 2015 for generating promising quality on image and audio, have recently revived interest in a text-to-image generation. First, the diffusion process starts by adding noise to the original images, gradually corrupting them in several steps. At each step, a small amount of noise is added to the images, resulting in progressively noisier images until they become indistinguishable from random noise. In addition, in reverse diffusion process aims to recover the original images from the noisy ones, denoising them step by step. A learned denoising function guides this process, typically implemented as a deep neural network. The training will minimize the difference between the denoised images and the original images at each step of the reverse diffusion process. The goal is to learn a function that can accurately denoise the images, effectively capturing the data distribution. To generate new images, the diffusion model starts with a random noise sample and applies the reverse diffusion process, denoising the sample step by step using the learned denoising function. While diffusion models have shown competitive quality compared to GANs and VAEs, they can be computationally expensive and slower during sampling due to requiring multiple forwards passes through the denoising function

## Challenges
Image generation, including GANs and diffusion models, has made significant advancements but still faces several limitations and challenges.

1. High-quality image generation: Although image generation models can create visually appealing images, generating high-resolution and photorealistic images remains challenging. High-quality image generation often requires larger and deeper neural networks, increasing computational cost and memory requirements. 
2. Model collapse: It can occur in generative models, particularly in the context of GANs. It happens when the generator produces limited or repetitive samples, failing to capture the full diversity of the data distribution. It can lead to reduced quality and diversity in generated images. 
3. Training instability: GANs, particularly, can be challenging due to their adversarial nature. Balancing the learning process between the generator and discriminator is crucial where model parameters oscillate, destabilize and never converge. It can lead to reduced quality and diversity in generated images. 
4. Precise control over generated content: Current image generation techniques often lack precise control over specific attributes or features in the generated images. Researchers are exploring techniques like conditional GANs and StyleGANs to improve control, but achieving fine-grained control and disentanglement of features remains challenging. 
5. Evaluation metrics: Evaluating generative models is important for assessing their ability to generate high-quality and diverse images. Existing evaluation metrics for unconditional generation include the Inception Score (IS) and Fréchet Inception Distance (FID), which measure quality and diversity based on the distribution of generated images. However, these metrics are not designed for the class-conditional setting, where the level at which the categorical condition manifests itself in the generated data needs to be assessed. The Classification Accuracy Score (CAS) has been proposed as an alternative, but it requires training a different classifier for each generative model, which can be biased and laborious. (Benny, Galanti, Benaim & Wolf, 2021). Therefore, Quantitatively evaluating the performance of generative models can be difficult due to the lack of a single, universally accepted metric. 
6. Generalization: Current image generation models often struggle to generalize to new or unseen data. It can limit their ability to generate images in different domains or with specific constraints without additional fine-tuning or retraining. 
7. Computational resources: Training state-of-the-art image generation models requires significant computational resources, which may not be accessible to all researchers and developers. It can lead to a concentration of research advancements among well-funded institutions and organizations. 
8. Ethical and Legal concerns: The potential for misuse of image generation technology raises ethical and legal concerns, such as generating deep fakes, creating inappropriate content, or infringing on copyrights. Addressing these issues requires technical solutions, policy, and regulatory measures.

Despite these challenges, image generation research continues to progress, and addressing these limitations will likely lead to even more impressive advances in the field.

## Conclusion
In conclusion, image generation has made significant progress in recent years, and various approaches have been developed to address different challenges. The advancements have led to the development of various categories of image generation, including image-to-image translation, sketch-to-image generation, text-to-image generation, layout-to-image generation, facial image generation, video generation, panoramic image generation, scene graph image generation, few-shot image generation, 3D image/object generation, and multi-view image generation. Notable contributions include:

- the use of diffusion models for video generation,
- utilizing CLIP Contrastive models for Text-Conditional, and
- combining transformer language models with high-fidelity diffusion models for textto-image synthesis.

Moreover, several deep-learning approaches to image generation have been extensively researched, such as Variational Autoencoders, Generative Adversarial Networks, and Diffusion models. However, image generation still faces challenges such as high-quality image generation, model collapse, training instability, precise control over generated content, evaluation metrics, generalization, computational resources, and ethical and legal concerns. Despite these challenges, image generation research continues to progress, and addressing these limitations will likely lead to even more impressive advances in the field.

## Reference
```
Agnese, J., Herrera, J., Tao, H., & Zhu, X. (2020). A survey and taxonomy of adversarialneural networks for text-to-image synthesis. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 10(4), e1345.

Benny, Y., Galanti, T., Benaim, S., & Wolf, L. (2021). Evaluation metrics for conditional image generation. International Journal of Computer Vision, 129, 1712-1731.

Chen, T., Cheng, M. M., Tan, P., Shamir, A., & Hu, S. M. (2009). Sketch2photo: Internet image montage. ACM transactions on graphics (TOG), 28(5), 1-10.

Chen, W., & Hays, J. (2018). Sketchygan: Towards diverse and realistic sketch to image synthesis. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 9416-9425).

Chen, X., Cohen-Or, D., Chen, B., & Mitra, N. J. (2020). Neural graphics pipeline for controllable image generation. arXiv preprint arXiv:2006.10569, 2.

Cheng, J., Liang, X., Shi, X., He, T., Xiao, T., & Li, M. (2023). LayoutDiffuse: Adapting Foundational Diffusion Models for Layout-to-Image Generation. arXiv preprint arXiv:2302.08908.

Cheng, S. I., Chen, Y. J., Chiu, W. C., Tseng, H. Y., & Lee, H. Y. (2023). AdaptivelyRealistic Image Generation from Stroke and Sketch with Diffusion Model. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 4054-4062).

Deng, Y., Yang, J., Chen, D., Wen, F., & Tong, X. (2020). Disentangled and controllable face image generation via 3d imitative-contrastive learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 5154-5163).

Eitz, M., Richter, R., Hildebrand, K., Boubekeur, T., & Alexa, M. (2011). Photosketcher: interactive sketch-based image synthesis. IEEE Computer Graphics and Applications, 31(6), 56-66.

Elasri, M., Elharrouss, O., Al-Maadeed, S., & Tairi, H. (2022). Image Generation: A Review. Neural Processing Letters, 54(5), 4609-4646.

Gao, C., Liu, Q., Xu, Q., Wang, L., Liu, J., & Zou, C. (2020). Sketchycoco: Image generation from freehand scene sketches. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 5174-5183).

Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative Adversarial Networks. arXiv preprint arXiv:1406.2661.

Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., ... & Salimans, T. (2022). Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303.

Kingma, D. P., & Welling, M. (2019). An introduction to variational
autoencoders. Foundations and Trends® in Machine Learning, 12(4), 307-392.

Kim, T., Cha, M., Kim, H., Lee, J. K., & Kim, J. (2017, July). Learning to discover crossdomain relations with generative adversarial networks. In International conference on machine learning (pp. 1857-1865). PMLR.

Lu, Y., Wu, S., Tai, Y. W., & Tang, C. K. (2018). Image generation from sketch constraint using contextual gan. In Proceedings of the European conference on computer vision (ECCV) (pp. 205-220).

Mao, X., Wang, S., Zheng, L., & Huang, Q. (2018). Semantic invariant cross-domain image generation with generative adversarial networks. Neurocomputing, 293, 55-63.

Mittal, G., Agrawal, S., Agarwal, A., Mehta, S., & Marwah, T. (2019). Interactive image generation using scene graphs. arXiv preprint arXiv:1905.03743.

Ojha, U., Li, Y., Lu, J., Efros, A. A., Lee, Y. J., Shechtman, E., & Zhang, R. (2021). Fewshot image generation via cross-domain correspondence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10743-10752).

Pan, J., Wang, C., Jia, X., Shao, J., Sheng, L., Yan, J., & Wang, X. (2019). Video generation from single semantic label map. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 3733-3742).

Pang, Y., Lin, J., Qin, T., & Chen, Z. (2021). Image-to-image translation: Methods and applications. IEEE Transactions on Multimedia, 24, 3859-3881.

Pathak, D., Krahenbuhl, P., Donahue, J., Darrell, T., & Efros, A. A. (2016). Context encoders: Feature learning by inpainting. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2536-2544).

Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M. (2022). Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125. 

Tang, H., Xu, D., Liu, G., Wang, W., Sebe, N., & Yan, Y. (2019, October). Cycle in cycle generative adversarial networks for keypoint-guided image generation. In Proceedings of the 27th ACM international conference on multimedia (pp. 2052-2060).

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E., ... & Norouzi, M. (2022). Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487.

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015, June). Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning (pp. 2256-2265). PMLR.

Song, D. Y., Lee, G., Lee, H., Um, G. M., & Cho, D. (2022, October). Weakly-Supervised Stitching Network for Real-World Panoramic Image Generation. In Computer Vision–ECCV 2022: 17th European Conference, Tel 

Aviv, Israel, October 23–27, 2022, Proceedings, Part XVI (pp. 54-71). Cham: Springer Nature Switzerland.

Tian, Y., Peng, X., Zhao, L., Zhang, S., & Metaxas, D. N. (2018). CR-GAN: learning complete representations for multi-view generation. arXiv preprint
arXiv:1806.11191.

Wang, H., Lin, G., Hoi, S. C., & Miao, C. (2022). 3D Cartoon Face Generation with Controllable Expressions from a Single GAN Image. arXiv preprint arXiv:2207.14425.

Xia, W., Yang, Y., Xue, J. H., & Wu, B. (2021). Tedigan: Text-guided diverse face image generation and manipulation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 2256-2265).

Zhao, B., Wu, X., Cheng, Z. Q., Liu, H., Jie, Z., & Feng, J. (2018, October). Multi-view image generation from a single-view. In Proceedings of the 26th ACM international conference on Multimedia (pp. 383-391).

Zhou, T., He, D., & Lee, C. H. (2020, April). Pixel-level bird view image generation from front view by using a generative adversarial network. In 2020 6th international conference on control, automation and robotics (ICCAR) (pp. 683-689). IEEE.

Zhu, X., Goldberg, A. B., Eldawy, M., Dyer, C. R., & Strock, B. (2007, July). A text-topicture synthesis system for augmenting communication. In AAAI (Vol. 7, pp.1590-1595).

Zhu, J. Y., Zhang, Z., Zhang, C., Wu, J., Torralba, A., Tenenbaum, J., & Freeman, B. (2018). Visual object networks: Image generation with disentangled 3D representations. Advances in neural information processing systems, 31.
```