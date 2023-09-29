import timm
import torch
import torchvision.transforms as transforms
from PIL import Image

class VitModel(torch.nn.Module):
    def __init__(self, model_name, num_classes):
        super(VitModel, self).__init__()
        self.modelVit = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        data_config = timm.data.resolve_model_data_config(self.modelVit)
        self.transforms = timm.data.create_transform(**data_config, is_training=True)

    def forward(self, x):
        d1 = self.modelVit(x)
        return d1

def load_model(model_path, model_name, num_classes):
    model = VitModel(model_name, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    x = transform(image)
    x = x.unsqueeze(0)
    return x

def main():
    model_path = 'modelTRANSFORMER_VIT' #Link to download the weights in README
    img_size = 224
    num_classes = 8

    model = load_model(model_path, 'vit_large_patch16_224.augreg_in21k_ft_in1k', num_classes)

    image_path = 'crops/00000325.jpg'
    x = preprocess_image(image_path, img_size)

    out, cls, attn = model(x)
    _, preds = torch.max(out, 1)

    classes = ['Domestic Cattle', 'Eurasian Badger', 'European Hare', 'Grey Wolf', 'Human', 'Red Deer', 'Red Fox', 'Wild Boar']
    predicted_class = classes[preds.cpu().numpy()[0]]
    confidence = torch.nn.functional.softmax(out, dim=1).max().item()

    print(f'Predicted Class: {predicted_class}')
    print(f'Confidence: {confidence:.2f}')

if __name__ == "__main__":
    main()
