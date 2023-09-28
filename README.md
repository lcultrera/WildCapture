# WildCapture
# Out Of Distribution Generalization in Computer Vision Workshop (ICCV 2023)

This repository contains the code and dataset used in the paper titled "Out Of Distribution Generalization in Computer Vision Workshop" published at ICCV 2023, Paris Out Of Distribution Generalization in Computer Vision workshop.

## Paper Abstract

[Out-of-Distribution (OOD) detection is a crucial challenge in computer vision, especially when deploying machine learning models in the real world. In this paper, we propose a novel OOD detection method leveraging Visual Attention Heatmaps from a Vision Transformer (ViT) classifier. Our approach involves training a Convolutional Autoencoder to reconstruct attention heatmaps produced by a ViT classifier, enabling accurate image reconstruction and effective OOD detection. Moreover, our method does not require additional labels during training, ensuring efficiency and ease of implementation. We validate our approach on a standard OOD benchmark using CIFAR10 and CIFAR100. To test OOD in a real-world setting we also collected a novel dataset: WildCapture. Our new dataset comprises more than 60k  wild animal shots, from 15 different wildlife species, taken via phototraps in varying lighting conditions. The dataset is fully annotated with animal bounding boxes and species.]

## Dataset

The dataset used in this work can be found at https://drive.google.com/drive/folders/1823UnVK94NYRTokxkzBbgeYRVL8YSPG4?usp=drive_link. Detailed instructions for data preparation and usage can be found in the `dataset/README.md` file.

## Code

The code for our experiments is organized in the `code/` directory. You can find detailed instructions on how to run the code and reproduce the results in the `code/README.md` file.

## Citation

If you find this work useful in your research, please consider citing: TBD