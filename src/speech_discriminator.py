import torch
from pytorch_model_summary import summary
from torch import Tensor
from torch.nn import Module, Linear, LeakyReLU, Dropout, Sequential
from gradient_reversal_layer import GradientReversal


class SpeechDiscriminator(Module):

    def __init__(self,
                 features_length: int,
                 alpha: Tensor) -> None:
        super().__init__()
        
        self.alpha = alpha
        self.grl = GradientReversal()

        self.block_1 = Sequential(
            Linear(in_features=features_length, out_features=int(features_length*0.75)),
            LeakyReLU(),
            Dropout(0.25)
        )

        self.block_2 = Sequential(
            Linear(in_features=int(features_length*0.75), out_features=int(features_length*0.5)),
            LeakyReLU(),
        )

        self.block_3 = Sequential(
            Linear(in_features=int(features_length*0.5), out_features=int(features_length*0.25)),
            LeakyReLU(),
        )

        self.block_4 = Sequential(
            Linear(in_features=int(features_length*0.25), out_features=1),
        )

    def forward(self, X: Tensor) -> Tensor:
        X = self.grl(X, self.alpha)
        X = self.block_1(X)
        X = self.block_2(X)
        X = self.block_3(X)
        y = self.block_4(X)
        return y
    
    def get_alpha(self) -> Tensor:
        return self.alpha
    
    def set_alpha(self, new_alpha: Tensor) -> None:
        """Set a new domain adaptation parameter
        :param new_alpha: New domain adaptation parameter
        :type new_alpha: int
        """
        self.alpha = new_alpha