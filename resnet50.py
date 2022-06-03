import torchvision.models as models

if __name__ == "__main__":
    resnet50 = models.resnet50(pretrained=True)

    print(resnet50)