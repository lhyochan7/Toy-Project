import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np


class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == "__main__":
    img = cv2.imread('painting.PNG')
    #img = img.transpose(2,0,1).astype(np.float32)[None]
    img = cv2.resize(img, (224,224))

    print(img.shape)

    cv2.imshow('img', img)
    cv2.waitKey(100)

    transform = transforms.ToTensor()
    img = transform(img)


    img = img.reshape(1,3,224,224)
    print(img.shape)

    res50_model = models.resnet50(pretrained=True)
    res50_conv2 = ResNet50Bottom(res50_model)

    outputs = res50_conv2(img)
    print(outputs.data.shape)  # => torch.Size([4, 2048, 7, 7])

