import torch
import torch.nn as nn
import timm

class VitModel(nn.Module):
    def __init__(self, num_classes):
        super(VitModel, self).__init__()

        # Create the ViT model
        self.model_vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        self.transforms = timm.data.transforms_presets.vit_base_patch16_224_in21k

    def forward(self, x):
        # Apply data transformations
        x = self.transforms(x)
        
        # Pass the input through the ViT model
        output = self.model_vit(x)

        return output

