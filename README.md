# WildCapture

This repository contains the code and dataset used in the paper titled "Leveraging Visual Attention for out-of-distribution Detection" published at ICCV 2023, Paris Out Of Distribution Generalization in Computer Vision workshop.

## Paper Abstract

[Out-of-Distribution (OOD) detection is a crucial challenge in computer vision, especially when deploying machine learning models in the real world. In this paper, we propose a novel OOD detection method leveraging Visual Attention Heatmaps from a Vision Transformer (ViT) classifier. Our approach involves training a Convolutional Autoencoder to reconstruct attention heatmaps produced by a ViT classifier, enabling accurate image reconstruction and effective OOD detection. Moreover, our method does not require additional labels during training, ensuring efficiency and ease of implementation. We validate our approach on a standard OOD benchmark using CIFAR10 and CIFAR100. To test OOD in a real-world setting we also collected a novel dataset: WildCapture. Our new dataset comprises more than 60k  wild animal shots, from 15 different wildlife species, taken via phototraps in varying lighting conditions. The dataset is fully annotated with animal bounding boxes and species.]


## Table of Contents

- [Usage](#usage)
  - [Training the VIT Classifier] To train the ViT classifier run the train.py script in "train_classifier" folder. (preTrained models provided )
  - [Testing the VIT Classifier]To test the ViT classifier run the test_classifier.py script in "test_classifier" folder.
  - [Extracting Attention Heatmaps] To extract Attention Heatmpas from a preTrained VitModel run the test classifier_script and make sure that "extract_attention"
"save_att_heatmap" parameter are both setted at True in the config file "Test_Classifier/config/config.yaml"
  - [Training the Convolutional Autoencoder (CAE)] To train the CAE run the train.py script in "AutoEncoder" folder. (preTrained models provided )
  - [Testing the Autoencoder and Calculating Reconstruction Error] To test the Autoencoder run the test_autoencoder.py script in "AutoEncoder" folder. In the Test_config file make sure to set correctly the parameters "dest_file" (pkl dest file where the code save the reconstruction error for each image)  and "model_path"(path to pretrained CAE model)

## Dataset

The dataset used in this work can be found at https://drive.google.com/drive/folders/1823UnVK94NYRTokxkzBbgeYRVL8YSPG4?usp=drive_link. As mentioned in the paper the dataset can be splitted to perform OOD detection tasks. All the pre-trained models provided in this repo refer to the following dataset split:

In distribution: Domestic Cattle, Eurasian Badger, European Hare, Grey Wolf, Red Deer, Red Fox, Wild Boar

Out-Of-Distribution: Beech Marten, Crested Porcupine, Domestic Dog, Domestic Horse, European Roe Deer, Persian Fallow Deer, Western Polecat, Wild Cat


## PreTrained Models

You can download the pre-trained classifier weights for the model from the following link: https://drive.google.com/file/d/1sLRQfSZ03ByHjpkOIYxQSIsslHlm-Pvn/view?usp=sharing
You can find the pretrained weights for the Convolutional AutoEncoder in the folder AutoEncoder/AutoEncoder_Weights

### Instructions

1. Click on the "Download Classifier Weights" link above.
2. Save the downloaded file to a location on your local machine.
3. You can then load the weights into your model using the provided code in the `test_classifier.py` script.

```python
import torch
# Load the pre-trained classifier weights
model_weights_path = 'path_to_downloaded_weights'
model = YourModel()
model.load_state_dict(torch.load(model_weights_path))
model.eval()
## Code
```

If you find this work useful in your research, please consider citing: TBD


