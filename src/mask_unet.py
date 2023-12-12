from typing import Union, Tuple, Optional
import torch
from torch import Tensor
from pytorch_model_summary import summary
import torch.nn as nn


class MaskUnet(nn.Module):
    def __init__(self,
                conv1_output_dim: int,
                conv2_output_dim: int,
                conv3_output_dim: int,
                conv4_output_dim: int,
                conv5_output_dim: int,
                conv6_output_dim: int,
                conv1_kernel_size: Union[int, Tuple[int, int]],
                conv2_kernel_size: Union[int, Tuple[int, int]],
                conv3_kernel_size: Union[int, Tuple[int, int]],
                conv4_kernel_size: Union[int, Tuple[int, int]],
                conv5_kernel_size: Union[int, Tuple[int, int]],
                conv6_kernel_size: Union[int, Tuple[int, int]],
                dropout: Optional[float] = 0.5,
                conv1_stride: Optional[Union[int, Tuple[int, int]]] = 2,
                conv2_stride: Optional[Union[int, Tuple[int, int]]] = 2,
                conv3_stride: Optional[Union[int, Tuple[int, int]]] = 2,
                conv4_stride: Optional[Union[int, Tuple[int, int]]] = 2,
                conv5_stride: Optional[Union[int, Tuple[int, int]]] = 2,
                conv6_stride: Optional[Union[int, Tuple[int, int]]] = 2,
                conv1_padding: Optional[Union[int, Tuple[int, int]]] = 2,
                conv2_padding: Optional[Union[int, Tuple[int, int]]] = 2,
                conv3_padding: Optional[Union[int, Tuple[int, int]]] = 2,
                conv4_padding: Optional[Union[int, Tuple[int, int]]] = 2,
                conv5_padding: Optional[Union[int, Tuple[int, int]]] = 2,
                conv6_padding: Optional[Union[int, Tuple[int, int]]] = 2) -> None:
        super().__init__()

        self.encoder1 = nn.Sequential()
        self.encoder1.add_module('conv1', nn.Conv2d(1, 
                                                    conv1_output_dim, 
                                                    conv1_kernel_size,
                                                    conv1_stride,
                                                    conv1_padding))
        self.encoder1.add_module('en_batchnorm1', nn.BatchNorm2d(conv1_output_dim))
        self.encoder1.add_module('leakyRelu1', nn.LeakyReLU(0.2))

        self.encoder2 = nn.Sequential()
        self.encoder2.add_module('conv2', nn.Conv2d(conv1_output_dim, 
                                                    conv2_output_dim, 
                                                    conv2_kernel_size,
                                                    conv2_stride,
                                                    conv2_padding))
        self.encoder2.add_module('en_batchnorm2', nn.BatchNorm2d(conv2_output_dim))
        self.encoder2.add_module('leakyRelu2', nn.LeakyReLU(0.2))

        self.encoder3 = nn.Sequential()
        self.encoder3.add_module('conv3', nn.Conv2d(conv2_output_dim, 
                                                    conv3_output_dim, 
                                                    conv3_kernel_size,
                                                    conv3_stride,
                                                    conv3_padding))
        self.encoder3.add_module('en_batchnorm3', nn.BatchNorm2d(conv3_output_dim))
        self.encoder3.add_module('leakyRelu3', nn.LeakyReLU(0.2))

        self.encoder4 = nn.Sequential()
        self.encoder4.add_module('conv4', nn.Conv2d(conv3_output_dim, 
                                                    conv4_output_dim, 
                                                    conv4_kernel_size,
                                                    conv4_stride,
                                                    conv4_padding))
        self.encoder4.add_module('en_batchnorm4', nn.BatchNorm2d(conv4_output_dim))
        self.encoder4.add_module('leakyRelu4', nn.LeakyReLU(0.2))

        self.encoder5 = nn.Sequential()
        self.encoder5.add_module('conv5', nn.Conv2d(conv4_output_dim,
                                                    conv5_output_dim,
                                                    conv5_kernel_size,
                                                    conv5_stride,
                                                    conv5_padding))
        self.encoder5.add_module('en_batchnorm5', nn.BatchNorm2d(conv5_output_dim))
        self.encoder5.add_module('leakyRelu5', nn.LeakyReLU(0.2))

        self.encoder6 = nn.Sequential()
        self.encoder6.add_module('conv6', nn.Conv2d(conv5_output_dim,
                                                    conv6_output_dim,
                                                    conv6_kernel_size,
                                                    conv6_stride,
                                                    conv6_padding))
        self.encoder6.add_module('en_batchnorm6', nn.BatchNorm2d(conv6_output_dim))
        self.encoder6.add_module('leakyRelu6', nn.LeakyReLU(0.2))

        self.decoder1 = nn.ConvTranspose2d(conv6_output_dim,
                                           conv5_output_dim,
                                           conv6_kernel_size,
                                           conv6_stride,
                                           conv6_padding)
        self.decoder1_other = nn.Sequential()
        self.decoder1_other.add_module('de_batchnorm1', nn.BatchNorm2d(conv5_output_dim))
        self.decoder1_other.add_module('relu1', nn.ReLU())
        self.decoder1_other.add_module('dropout1', nn.Dropout2d(dropout))

        self.decoder2 = nn.ConvTranspose2d(conv5_output_dim*2,
                                           conv4_output_dim,
                                           conv5_kernel_size,
                                           conv5_stride,
                                           conv5_padding)
        self.decoder2_other = nn.Sequential()
        self.decoder2_other.add_module('de_batchnorm2', nn.BatchNorm2d(conv4_output_dim))
        self.decoder2_other.add_module('relu2', nn.ReLU())
        self.decoder2_other.add_module('dropout2', nn.Dropout2d(dropout))

        self.decoder3 = nn.ConvTranspose2d(conv4_output_dim*2,
                                           conv3_output_dim,
                                           conv4_kernel_size,
                                           conv4_stride,
                                           conv4_padding)
        self.decoder3_other = nn.Sequential()
        self.decoder3_other.add_module('de_batchnorm3', nn.BatchNorm2d(conv3_output_dim))
        self.decoder3_other.add_module('relu3', nn.ReLU())
        self.decoder3_other.add_module('dropout3', nn.Dropout2d(dropout))

        self.decoder4 = nn.ConvTranspose2d(conv3_output_dim*2,
                                           conv2_output_dim,
                                           conv3_kernel_size,
                                           conv3_stride,
                                           conv3_padding)
        self.decoder4_other = nn.Sequential()
        self.decoder4_other.add_module('de_batchnorm4', nn.BatchNorm2d(conv2_output_dim))
        self.decoder4_other.add_module('relu4', nn.ReLU())

        self.decoder5 = nn.ConvTranspose2d(conv2_output_dim*2,
                                           conv1_output_dim,
                                           conv2_kernel_size,
                                           conv2_stride,
                                           conv2_padding)
        self.decoder5_other = nn.Sequential()
        self.decoder5_other.add_module('de_batchnorm5', nn.BatchNorm2d(conv1_output_dim))
        self.decoder5_other.add_module('relu5', nn.ReLU())

        self.decoder6 = nn.ConvTranspose2d(conv1_output_dim*2,
                                           1,
                                           conv1_kernel_size,
                                           conv1_stride,
                                           conv1_padding)
        self.decoder6_other = nn.Sequential()
        self.decoder6_other.add_module('sigmoid1',nn.Sigmoid())

    def forward(self, X: Tensor) -> Tensor:
        # input of shape (batchsize, channels, height, width)
        conv1_out = self.encoder1(X)
        conv2_out = self.encoder2(conv1_out)
        conv3_out = self.encoder3(conv2_out)
        conv4_out = self.encoder4(conv3_out)
        conv5_out = self.encoder5(conv4_out)
        conv6_out = self.encoder6(conv5_out)

        deconv1_out = self.decoder1(conv6_out, output_size = conv5_out.size())
        deconv1_out = self.decoder1_other(deconv1_out)

        deconv2_out = self.decoder2(torch.cat((conv5_out, deconv1_out), dim=1),
                                    output_size = conv4_out.size())
        deconv2_out = self.decoder2_other(deconv2_out)

        deconv3_out = self.decoder3(torch.cat((conv4_out, deconv2_out), dim=1),
                                    output_size = conv3_out.size())
        deconv3_out = self.decoder3_other(deconv3_out)

        deconv4_out = self.decoder4(torch.cat((conv3_out, deconv3_out), dim=1),
                                    output_size = conv2_out.size())
        deconv4_out = self.decoder4_other(deconv4_out)

        deconv5_out = self.decoder5(torch.cat((conv2_out, deconv4_out), dim=1),
                                    output_size = conv1_out.size())
        deconv5_out = self.decoder5_other(deconv5_out)

        deconv6_out = self.decoder6(torch.cat((conv1_out, deconv5_out), dim=1),
                                    output_size = X.size())
        mask = self.decoder6_other(deconv6_out)

        return mask


def main():
    model = MaskUnet(conv1_output_dim = 16,
                    conv2_output_dim = 32,
                    conv3_output_dim = 64,
                    conv4_output_dim = 128,
                    conv5_output_dim = 256,
                    conv6_output_dim = 512,
                    conv1_kernel_size = 5,
                    conv2_kernel_size = 5,
                    conv3_kernel_size = 5,
                    conv4_kernel_size = 5,
                    conv5_kernel_size = 5,
                    conv6_kernel_size = 5,
                    dropout = 0.5,
                    conv1_stride = 2,
                    conv2_stride = 2,
                    conv3_stride = 2,
                    conv4_stride = 2,
                    conv5_stride = 2,
                    conv6_stride = 2,
                    conv1_padding = 2,
                    conv2_padding = 2,
                    conv3_padding = 2,
                    conv4_padding = 2,
                    conv5_padding = 2,
                    conv6_padding = 2)
    print(summary(model, torch.rand(4,1,706,100)))


if __name__ == "__main__":
    main()




        
