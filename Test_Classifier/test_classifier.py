import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
from ForwardWrapper import forward_wrapper

class VitModel(torch.nn.Module):
    def __init__(self, model_name, num_classes, extract_attention):
        super(VitModel, self).__init__()
        self.modelVit = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        data_config = timm.data.resolve_model_data_config(self.modelVit)
        self.transforms = timm.data.create_transform(**data_config, is_training=True)
        
        if extract_attention == True:
                    self.modelVit.blocks[-1].attn.forward = forward_wrapper(self.modelVit.blocks[-1].attn)

    def forward(self, x):
        if extract_attention == True:
            cls_weight = self.modelVit.blocks[-1].attn.cls_attn_map
            cls_weight =  cls_weight.mean(dim=1)
            a = torch.empty(cls_weight.shape[0], 1)
            a[:,0] = cls_weight[:,165]
            cls = torch.cat((cls_weight, a.cuda()), 1)
            cls = cls.view(-1, 14, 14, 1).detach().cpu().numpy()
            #cls_resized = F.interpolate(cls(1, 1, 14, 14), (224, 224), mode='bilinear').view( 224, 224, 1)
            cls = np.squeeze(cls,axis=0)
            return d1, cls, attn_map
        else:
            d1 = self.modelVit(x)
            return d1

def load_model(model_path, model_name, num_classes, extract_attention):
    model = VitModel(model_name, num_classes, extract_attention)
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
    
    # Define the path to the YAML config file
    config_path = 'config/config.yaml'

    # Load the config file
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Extract parameters from the config
    num_classes = config["num_classes"] #7
    img_size = config["img_size"] #224
    model_path = config["model_path"] #modelTRANSFORMER_VIT
    extract_attention = config["extract_attention"] #False


    model = load_model(model_path, 'vit_large_patch16_224.augreg_in21k_ft_in1k', num_classes)

    image_path = 'crops/00000325.jpg' #open an image from crop folder
    x = preprocess_image(image_path, img_size)

    out, cls, attn = model(x)
    _, preds = torch.max(out, 1)

    classes = ['Domestic Cattle', 'Eurasian Badger', 'European Hare', 'Grey Wolf', 'Red Deer', 'Red Fox', 'Wild Boar']
    predicted_class = classes[preds.cpu().numpy()[0]]
    confidence = torch.nn.functional.softmax(out, dim=1).max().item()

    print(f'Predicted Class: {predicted_class}')
    print(f'Confidence: {confidence:.2f}')

if __name__ == "__main__":
    main()
