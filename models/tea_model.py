import torch
import torch.nn as nn
import torchvision.models as models

# multi-task CNN model for tea grading
class TeaNet(nn.Module):
    # initialize the model
    def __init__(self, num_grades=3, num_qualities=10):
        super(TeaNet, self).__init__()

        # load pre-trained ResNet18 backbone
        self.backbone = models.resnet18(pretrained=True)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # output two heads (grade - quality)
        self.grade_head = nn.Linear(in_features, num_grades)
        self.quality_head = nn.Linear(in_features, num_qualities)

    # pass to the network
    def forward(self, x):
        # extract features using ResNet18 backbone
        features = self.backbone(x)

        # pass features through both heads
        grade_out = self.grade_head(features)
        quality_out = self.quality_head(features)
        
        return grade_out, quality_out
