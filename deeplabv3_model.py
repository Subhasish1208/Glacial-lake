import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def get_deeplabv3_model(pretrained=True, num_classes=1):
    if pretrained:
        # Load model with pre-trained ResNet50 backbone (COCO weights)
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights)
        
        # Modify the classifier head for binary segmentation (1 output channel)
        # DeepLabV3 classifier structure: FCNHead -> Sequential(Conv2d, BatchNorm2d, ReLU, Dropout, Conv2d)
        in_channels = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
        # Modify the auxiliary classifier if it exists
        if model.aux_classifier is not None:
            in_channels_aux = model.aux_classifier[4].in_channels
            model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=1)
    else:
        # Train from scratch
        model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
        
    return model

if __name__ == '__main__':
    # Simple unit test
    model = get_deeplabv3_model(pretrained=True)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)['out']
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    assert out.shape == (2, 1, 512, 512), f"Expected shape (2, 1, 512, 512), but got {out.shape}"
    print("DeepLabV3 model loaded and tested successfully!")
