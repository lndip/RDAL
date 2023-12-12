import torch
from pytorch_model_summary import summary
from torch import Tensor
from typing import Tuple, Union, Optional
from torch.nn import Module, Conv2d, Sequential, BatchNorm2d, Dropout2d, Linear, MaxPool2d
from torch.nn.modules.activation import ReLU


class FeatureExtractor(Module):
    def __init__(self,
                conv_1_output_dim: int,
                conv_2_output_dim: int,
                conv_3_output_dim: int,
                conv_4_output_dim: int,
                conv_1_kernel_size: Union[int, Tuple[int, int]],
                conv_2_kernel_size: Union[int, Tuple[int, int]],
                conv_3_kernel_size: Union[int, Tuple[int, int]],
                conv_4_kernel_size: Union[int, Tuple[int, int]],
                pooling_1_kernel_size: Union[int, Tuple[int, int]],
                pooling_2_kernel_size: Union[int, Tuple[int, int]],
                pooling_3_kernel_size: Union[int, Tuple[int, int]],
                global_pooling_kernel_size: Union[int, Tuple[int, int]],
                output_feature_length: int,
                dropout: Optional[float] = 0.25,
                conv_1_stride: Optional[Union[int, Tuple[int, int]]] = 1,
                conv_2_stride: Optional[Union[int, Tuple[int, int]]] = 1,
                conv_3_stride: Optional[Union[int, Tuple[int, int]]] = 1,
                conv_4_stride: Optional[Union[int, Tuple[int, int]]] = 1,
                conv_1_padding: Optional[Union[int, Tuple[int, int]]] = 0,
                conv_2_padding: Optional[Union[int, Tuple[int, int]]] = 0,
                conv_3_padding: Optional[Union[int, Tuple[int, int]]] = 0,
                conv_4_padding: Optional[Union[int, Tuple[int, int]]] = 0,
                pooling_1_stride: Optional[Union[int, Tuple[int, int]]] = None,
                pooling_2_stride: Optional[Union[int, Tuple[int, int]]] = None,
                pooling_3_stride: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> None:

        super().__init__()

        self.block_1 = Sequential(
            Conv2d(in_channels=1,
                   out_channels=conv_1_output_dim,
                   kernel_size=conv_1_kernel_size,
                   stride=conv_1_stride,
                   padding=conv_1_padding),
            ReLU(),
            BatchNorm2d(num_features=conv_1_output_dim),
            MaxPool2d(kernel_size=pooling_1_kernel_size,
                      stride=pooling_1_stride),
            Dropout2d(dropout)
        )

        self.block_2 = Sequential(
            Conv2d(in_channels=conv_1_output_dim,
                   out_channels=conv_2_output_dim,
                   kernel_size=conv_2_kernel_size,
                   stride=conv_2_stride,
                   padding=conv_2_padding),
            ReLU(),
            BatchNorm2d(num_features=conv_2_output_dim),
            MaxPool2d(kernel_size=pooling_2_kernel_size,
                      stride=pooling_2_stride),
            Dropout2d(dropout)
        )

        self.block_3 = Sequential(
            Conv2d(in_channels=conv_2_output_dim,
                   out_channels=conv_3_output_dim,
                   kernel_size=conv_3_kernel_size,
                   stride=conv_3_stride,
                   padding=conv_3_padding),
            ReLU(),
            BatchNorm2d(num_features=conv_3_output_dim),
            MaxPool2d(kernel_size=pooling_3_kernel_size,
                      stride=pooling_3_stride),
            Dropout2d(dropout)
        )

        self.block_4 = Sequential(
            Conv2d(in_channels=conv_3_output_dim,
                   out_channels=conv_4_output_dim,
                   kernel_size=conv_4_kernel_size,
                   stride=conv_4_stride,
                   padding=conv_4_padding),
            ReLU(),
            BatchNorm2d(num_features=conv_4_output_dim),
            Dropout2d(dropout)
        )

        self.global_max_pooling = MaxPool2d(kernel_size=global_pooling_kernel_size, stride=None)
        self.fc = Linear(in_features=conv_4_output_dim, out_features=output_feature_length)
        self.conv_4_output_dim = conv_4_output_dim

        self.fc1 = Sequential(
            Linear(in_features=conv_4_output_dim, out_features=256), 
            ReLU()
        )
        self.fc2 =Sequential(
            Linear(in_features=256, out_features=128), 
            ReLU()
        )
        self.fc3 = Linear(in_features=128, out_features=output_feature_length)

    def forward(self, X: Tensor) -> Tensor:
        y = self.block_1(X)
        y = self.block_2(y)
        y = self.block_3(y)
        y = self.block_4(y)

        y = self.global_max_pooling(y)
        y = y.view(-1, self.conv_4_output_dim)
        y = self.fc(y)

        return y


def main():
    conv_1_output_dim = 64
    conv_2_output_dim = 128
    conv_3_output_dim = 256
    conv_4_output_dim = 512

    global_pooling_kernel_size = (4,8)
    output_feature_length = 64

    conv_1_kernel_size = 3
    conv_2_kernel_size = 3
    conv_3_kernel_size = 3
    conv_4_kernel_size = 3

    pooling_1_kernel_size = 2
    pooling_2_kernel_size = 2
    pooling_3_kernel_size = 2

    model = FeatureExtractor(
        conv_1_output_dim=conv_1_output_dim,
        conv_2_output_dim=conv_2_output_dim,
        conv_3_output_dim=conv_3_output_dim,
        conv_4_output_dim=conv_4_output_dim,
        conv_1_kernel_size=conv_1_kernel_size,
        conv_2_kernel_size=conv_2_kernel_size,
        conv_3_kernel_size=conv_3_kernel_size,
        conv_4_kernel_size=conv_4_kernel_size,
        pooling_1_kernel_size=pooling_1_kernel_size,
        pooling_2_kernel_size=pooling_2_kernel_size,
        pooling_3_kernel_size=pooling_3_kernel_size,
        global_pooling_kernel_size = global_pooling_kernel_size,
        output_feature_length=output_feature_length,
    )

    print(summary(model, torch.rand(4,1,64,100).float()))