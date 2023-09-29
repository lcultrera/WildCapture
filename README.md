# WildCapture

This repository contains the code and dataset used in the paper titled "Leveraging Visual Attention for out-of-distribution Detection" published at ICCV 2023, Paris Out Of Distribution Generalization in Computer Vision workshop.

## Paper Abstract

[Out-of-Distribution (OOD) detection is a crucial challenge in computer vision, especially when deploying machine learning models in the real world. In this paper, we propose a novel OOD detection method leveraging Visual Attention Heatmaps from a Vision Transformer (ViT) classifier. Our approach involves training a Convolutional Autoencoder to reconstruct attention heatmaps produced by a ViT classifier, enabling accurate image reconstruction and effective OOD detection. Moreover, our method does not require additional labels during training, ensuring efficiency and ease of implementation. We validate our approach on a standard OOD benchmark using CIFAR10 and CIFAR100. To test OOD in a real-world setting we also collected a novel dataset: WildCapture. Our new dataset comprises more than 60k  wild animal shots, from 15 different wildlife species, taken via phototraps in varying lighting conditions. The dataset is fully annotated with animal bounding boxes and species.]

## Dataset

The dataset used in this work can be found at https://drive.google.com/drive/folders/1823UnVK94NYRTokxkzBbgeYRVL8YSPG4?usp=drive_link. Detailed instructions for data preparation and usage can be found in the `dataset/README.md` file.

## PreTrained Weights

You can download the pre-trained classifier weights for the model from the following link: https://drive.google.com/file/d/1sLRQfSZ03ByHjpkOIYxQSIsslHlm-Pvn/view?usp=sharing

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

The code for our experiments is organized as follow: 

WildCapture/

Train_Vit_Classifier/
│

├── dataloader/

│ ├── dataloader.py

│ └── datasetSplitter.py

│

├── config/

│ ├── config.yaml

|

├── loss_optimizer/

│ ├── loss_optimizer.py

│

├── model/

│ ├── VitModel.py

│

└── train.py #train the vit classifier 

## Citation

If you find this work useful in your research, please consider citing: TBD
