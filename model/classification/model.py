# Standard Library Modules
import os
import sys
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torchvision
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class ClassificationModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(ClassificationModel, self).__init__()
        self.args = args

        if args.model_type == 'vgg11':
            self.model = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.IMAGENET1K_V1)
            args.image_resize_size = 224 # VGG11 requires 224x224 image size

            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_output_size = self.model.classifier[0].in_features
        elif args.model_type == 'resnet50':
            self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            args.image_resize_size = 224 # ResNet50 requires 224x224 image size

            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_output_size = self.model.fc.in_features
        elif args.model_type == 'resnet152':
            self.model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
            args.image_resize_size = 224 # ResNet152 requires 224x224 image size

            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_output_size = self.model.fc.in_features
        elif args.model_type == 'efficientnet_b0':
            self.model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            args.image_resize_size = 224 # EfficientNet-B0 requires 224x224 image size

            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_output_size = self.model.classifier[1].in_features
        elif args.model_type == 'efficientnet_b7':
            self.model = torchvision.models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1)
            args.image_resize_size = 600 # EfficientNet-B7 requires 600x600 image size

            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_output_size = self.model.classifier[1].in_features
        elif args.model_type == 'vit_b_16':
            self.model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
            args.image_resize_size = 224 # ViT-B/16 requires 224x224 image size

            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_output_size = self.model.heads[0].in_features
        else:
            raise NotImplementedError(f'Invalid model type: {args.model_type}')

        # Define the final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_output_size, self.feature_output_size // 2),
            nn.Dropout(self.args.dropout_rate),
            nn.ReLU(),
            nn.Linear(self.feature_output_size // 2, args.num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if 'vit' in self.args.model_type:
            # vit requires preprocessing
            encoder = self.feature_extractor[1]
            processed_img = self.model._process_input(images)

            n = processed_img.size(0)
            # Expand the class token to the full batch
            batch_class_token = self.model.class_token.expand(n, -1, -1)
            processed_img = torch.cat([batch_class_token, processed_img], dim=1)

            encoded_img = encoder(processed_img)
            features = encoded_img[:, 0]
        elif self.args.model_type in ['vgg11', 'resnet50', 'resnet152', 'efficientnet_b0', 'efficientnet_b7']:
            features = self.feature_extractor(images)

        features = features.view(features.size(0), -1) # Flatten the features to (batch_size, feature_output_size)
        logits = self.classifier(features)

        return logits
