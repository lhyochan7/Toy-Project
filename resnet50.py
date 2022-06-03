import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == "__main__":

    inputs = torch.rand(1, 3, 224, 224)
    res50_model = models.resnet50(pretrained=True)
    res50_conv2 = ResNet50Bottom(res50_model)

    outputs = res50_conv2(inputs)
    print(outputs.data.shape)  # => torch.Size([4, 2048, 7, 7])

