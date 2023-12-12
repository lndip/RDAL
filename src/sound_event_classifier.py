import torch
from pytorch_model_summary import summary
from torch import Tensor
from torch.nn import Module, Linear


class SoundEventClassifier(Module):

    def __init__(self,
                 features_length: int,
                 num_class: int) -> None:
        super().__init__()

        self.fc = Linear(in_features=features_length, out_features=num_class)

    def forward(self, X: Tensor) -> Tensor:
        y = self.fc(X)

        return y