import timm
import torch


class StudentNetwork(torch.nn.Module):
    def __init__(self, model_name="timm/tinynet_a"):
        super().__init__()
        self.cnn = timm.create_model(
            model_name=model_name,
            pretrained=True,
            num_classes=0,
        )

        self.fc = torch.nn.Linear(self.cnn.num_features, 512)

    def forward(self, input):
        feat = self.cnn(input)
        out = self.fc(feat)
        return out

    def get_embedding(self, input):
        return self.cnn(input)

    def set_output_classes(self, num_classes):
        self.fc = torch.nn.Linear(self.cnn.num_features, num_classes)
