import os
import torch
import numpy as np
import random

from pathlib import Path
from typing import Union

from feature_extractor import FeatureExtractor
from sound_event_classifier import SoundEventClassifier
from speech_discriminator import SpeechDiscriminator
from verifying_sd import VerifyingSD
from mask_unet import MaskUnet
from verifying_gd import VerifyingGD

from parameters import *

# Global constants
FREESOUND_LABELS = ['dog_barking', 'glass_breaking', 'gun_shot', 'cough', \
                    'slam', 'applause', 'dishes_pot_pan', 'toilet_flush', \
                    'cat_meowing', 'doorbell', 'crying', 'drill']
SPEECH_LABELS = ['no_speech', 'speech']
GENDER_LABELS = ['female', 'male']

MODELS_DIR = Path("RDAL","models")

NPY_DATA_SPLIT = Path("RDAL", "data_final", "final_data_npy_split")
FREESOUND_DIR = Path("RDAL", "data_final", "processed_freesound_data")
PICKLE_DATA = Path("RDAL", "pickle_data")
PICKLE_RESULTS = Path("RDAL", "pickle_results")


def get_files_from_dir_with_pathlib(dir_name: Union[str, Path]) -> 'list[Path]':
    """Returns the files in the directory `dir_name` using the pathlib package.
    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[Path]
    """
    return sorted(list(Path(dir_name).iterdir()))


def empty_dir(dir_path: Path):
    """Erase all the files inside the directory.
    :param dir_path: Path of the directory.
    :type dir_path: Path
    """
    filepaths = get_files_from_dir_with_pathlib(dir_path)
    for f in filepaths:
        os.remove(f)


def setting_seed(seed:int):
    """Set the seed for the random number generator. This is for reproducibility and debugging.

    :param seed: The seed to be set
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_models(device:str) -> tuple:
    """Generate the models

    :param device: The device to run the models on
    :type device: str
    :return: The models
    :rtype: tuple
    """
    feature_extractor = FeatureExtractor(
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
        global_pooling_kernel_size=global_pooling_kernel_size,
        output_feature_length=output_feature_length,
    ).to(device)

    mask_unet = MaskUnet(conv1_output_dim = unet_conv1_output_dim,
                        conv2_output_dim = unet_conv2_output_dim,
                        conv3_output_dim = unet_conv3_output_dim,
                        conv4_output_dim = unet_conv4_output_dim,
                        conv5_output_dim = unet_conv5_output_dim,
                        conv6_output_dim = unet_conv6_output_dim,
                        conv1_kernel_size = unet_conv1_kernel_size,
                        conv2_kernel_size = unet_conv2_kernel_size,
                        conv3_kernel_size = unet_conv3_kernel_size,
                        conv4_kernel_size = unet_conv4_kernel_size,
                        conv5_kernel_size = unet_conv5_kernel_size,
                        conv6_kernel_size = unet_conv6_kernel_size,
                        dropout = unet_dropout,
                        conv1_stride = unet_conv1_stride,
                        conv2_stride = unet_conv2_stride,
                        conv3_stride = unet_conv3_stride,
                        conv4_stride = unet_conv4_stride,
                        conv5_stride = unet_conv5_stride,
                        conv6_stride = unet_conv6_stride,
                        conv1_padding = unet_conv1_padding,
                        conv2_padding = unet_conv2_padding,
                        conv3_padding = unet_conv3_padding,
                        conv4_padding = unet_conv4_padding,
                        conv5_padding = unet_conv5_padding,
                        conv6_padding = unet_conv6_padding).to(device)

    sound_event_classifier = SoundEventClassifier(
        features_length=output_feature_length,
        num_class=num_class
    ).to(device)

    speech_discriminator = SpeechDiscriminator(
        features_length=output_feature_length,
        alpha=torch.tensor([1]).to(device)
    ).to(device)

    verifying_sd = VerifyingSD(
        features_length=output_feature_length
    ).to(device)
    
    verifying_gd = VerifyingGD(features_length=output_feature_length).to(device)

    return feature_extractor, mask_unet, sound_event_classifier, speech_discriminator, verifying_sd, verifying_gd


def calculate_grl(epochs:int, 
                  num_zeros_alpha:int, 
                  gamma:float) -> np.ndarray:
    """Calculate the GRL parameter

    :param epochs: Number of epochs
    :type epochs: int
    :param num_zeros_alpha: Number of epochs that GRL value equals to 0
    :type num_zeros_alpha: int
    :param gamma: Parameter to tune the GRL curve
    :type gamma: float
    :return: GRL parameter
    :rtype: np.ndarray
    """

    grl_parameter = np.zeros(epochs)
    grl_parameter[:num_zeros_alpha] = 0

    p = np.linspace(0,1, epochs-num_zeros_alpha)
    grl_parameter[num_zeros_alpha:] = 2/(1+np.exp(-gamma*p)) - 1
    return grl_parameter