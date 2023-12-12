
from forward_backward_pass import forward_backward_pass_rdal
from getting_and_init_the_data import get_all_data_loaders
from utils import (get_models, 
                   MODELS_DIR,
                   calculate_grl)
from verification import verification
from verification_gender import verification_gender

import torch

from torch.optim import SGD
from pathlib import Path
import numpy as np

def train_naive_rdal(batch_size:int, 
                training:str, 
                do_grad_clip:bool, 
                grad_clip_value:float, 
                patience:int, 
                job_idx:int, 
                device:str, 
                epochs:int) -> tuple:
    """Train the Naive RDAL

    :param batch_size: Batch size
    :type batch_size: int
    :param training: The specified training mode
    :type training: str
    :param do_grad_clip: Do gradient clipping?
    :type do_grad_clip: bool
    :param grad_clip_value: Gradient clipping value
    :type grad_clip_value: float
    :param patience: The patience for the early stopping
    :type patience: int
    :param job_idx: The job index on the cluster
    :type job_idx: int
    :param device: The current device
    :type device: str
    :param epochs: Number of training epochs
    :type epochs: int
    :return: The training and validation metrics
    :rtype: tuple
    """

    # Get the models
    feature_extractor, _, sound_event_classifier, speech_discriminator, _,_ = get_models(device)

    # Get the data loaders
    train_dataloader, val_dataloader, test_dataloader = get_all_data_loaders(batch_size)

    optimizer = SGD(list(feature_extractor.parameters()) +
                        list(sound_event_classifier.parameters()) + 
                        list(speech_discriminator.parameters()), lr=1e-2, momentum=0.9)
    
    # Domain adapatation parameters
    alphas = calculate_grl(epochs=epochs, num_zeros_alpha=30, gamma=100)

    # Variables for the early stopping
    highest_val_loss = 0
    best_epoch = 0
    patience_counter = 0

    # Each epoch's numbers
    epoch_train_loss_sec = []
    epoch_train_loss_sd = []
    epoch_val_loss_sec = []
    epoch_val_loss_sd = []
    epoch_train_event_accuracy = []
    epoch_train_speech_accuracy = []
    epoch_train_speech_recall = []
    epoch_val_event_accuracy = []
    epoch_val_speech_accuracy = []
    epoch_val_speech_recall = []

    grl_parameters = []

    for epoch in range(epochs):
        # Set the domain adaptation parameter
        alpha = torch.tensor([alphas[epoch]]).to(device)
        speech_discriminator.set_alpha(alpha)

        # Train the models
        feature_extractor, sound_event_classifier, speech_discriminator, train_loss_sec, train_loss_sd, train_event_accuracy, train_speech_accuracy, train_speech_recall, _, _ = \
            forward_backward_pass_rdal(feature_extractor,
                                    sound_event_classifier,
                                    speech_discriminator,
                                    train_dataloader,
                                    optimizer,
                                    device,
                                    do_grad_clip,
                                    grad_clip_value)

        # Validate the models
        with torch.no_grad():
            feature_extractor, sound_event_classifier, speech_discriminator, val_loss_sec, val_loss_sd, val_event_accuracy, val_speech_accuracy, val_speech_recall, _, _ = \
                forward_backward_pass_rdal(feature_extractor,
                                        sound_event_classifier,
                                        speech_discriminator,
                                        val_dataloader,
                                        None,
                                        device,
                                        False,
                                        None)

        # Append the numbers of the epoch
        epoch_train_loss_sec.append(train_loss_sec)
        epoch_train_loss_sd.append(train_loss_sd)
        epoch_val_loss_sec.append(val_loss_sec)
        epoch_val_loss_sd.append(val_loss_sd)
        epoch_train_event_accuracy.append(train_event_accuracy)
        epoch_train_speech_accuracy.append(train_speech_accuracy)
        epoch_train_speech_recall.append(train_speech_recall)
        epoch_val_event_accuracy.append(val_event_accuracy)
        epoch_val_speech_accuracy.append(val_speech_accuracy)
        epoch_val_speech_recall.append(val_speech_recall)
        grl_parameters.append(speech_discriminator.get_alpha().item())

        # Print the losses
        print(f'Epoch: {epoch:03d} | '
              f'GRL parameter: {alpha.item():7.4f} | '
              f'Mean training loss - SEC: {epoch_train_loss_sec[-1]:7.4f} | '
              f'Mean validation loss - SEC: {epoch_val_loss_sec[-1]:7.4f} | '
              f'Mean training loss - SD: {epoch_train_loss_sd[-1]:7.4f} | '
              f'Mean validation loss - SD: {epoch_val_loss_sd[-1]:7.4f}')

        # Verification process
        if(alpha.item() >= 0.99):
            if val_loss_sd > highest_val_loss:
                highest_val_loss = val_loss_sd
                best_epoch = epoch

                # Save the models
                torch.save(feature_extractor.state_dict(), Path(MODELS_DIR, training, f"{job_idx}", f"best_feature_extractor_{best_epoch}.pt"))
                torch.save(sound_event_classifier.state_dict(), Path(MODELS_DIR, training, f"{job_idx}", f"best_sound_event_classifier_{best_epoch}.pt"))
                torch.save(speech_discriminator.state_dict(), Path(MODELS_DIR, training, f"{job_idx}", f"best_speech_discriminator_{best_epoch}.pt"))

            # Early stopping
            elif epoch-best_epoch > patience:
                print(f'Early stopping criterion met at epoch {best_epoch} | '
                    f'highest validation SD loss: {highest_val_loss:7.4f}')
                break    

    # Return the losses, metrics, and other params
    return epoch_train_loss_sec, epoch_train_loss_sd, epoch_val_loss_sec, epoch_val_loss_sd, \
            epoch_train_event_accuracy, epoch_train_speech_accuracy, epoch_train_speech_recall, \
            epoch_val_event_accuracy, epoch_val_speech_accuracy, epoch_val_speech_recall, \
            grl_parameters, best_epoch


def test_naive_rdal(batch_size:int, 
                   training:str, 
                   job_idx:int, 
                   ver_epochs:int, 
                   ver_patience:int, 
                   device:str, 
                   best_epoch:int) -> tuple:
    """Test the Naive RDAL

    :param batch_size: Batch size
    :type batch_size: int
    :param training: The specified training mode
    :type training: str
    :param job_idx: The job index on the cluster
    :type job_idx: int
    :param ver_epochs: Number of epochs for the verification process
    :type ver_epochs: int
    :param ver_patience: Patience for the verification process
    :type ver_patience: int
    :param device: The current device
    :type device: str
    :param best_epoch: Best epoch for the models from the training
    :type best_epoch: int
    :return: The testing metrics
    :rtype: tuple
    """

    # Get the models
    feature_extractor, _, sound_event_classifier, speech_discriminator, _,_ = get_models(device)
    
    feature_extractor.load_state_dict(torch.load(Path(MODELS_DIR, training, f"{job_idx}", f"best_feature_extractor_{best_epoch}.pt"), map_location=device))
    sound_event_classifier.load_state_dict(torch.load(Path(MODELS_DIR, training, f"{job_idx}", f"best_sound_event_classifier_{best_epoch}.pt"), map_location=device))
    speech_discriminator.load_state_dict(torch.load(Path(MODELS_DIR, training, f"{job_idx}", f"best_speech_discriminator_{best_epoch}.pt"), map_location=device))

    print(f'Loaded feature extractor from {Path(MODELS_DIR, training, f"{job_idx}", f"best_feature_extractor_{best_epoch}.pt")}')
    print(f'Loaded sound event classifier from {Path(MODELS_DIR, training, f"{job_idx}", f"best_sound_event_classifier_{best_epoch}.pt")}')
    print(f'Loaded speech discriminator from {Path(MODELS_DIR, training, f"{job_idx}", f"best_speech_discriminator_{best_epoch}.pt")}')
    
    # Get the dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_all_data_loaders(batch_size)

    # Test the models on SEC and SD
    with torch.no_grad():
        _,_,_, test_loss_sec, test_loss_sd, test_event_accuracy, test_speech_accuracy, test_speech_recall, test_event_targets, test_event_predictions = \
            forward_backward_pass_rdal(feature_extractor,
                                    sound_event_classifier,
                                    speech_discriminator,
                                    test_dataloader,
                                    None,
                                    device,
                                    False,
                                    None)
    # Print test results
    print(f'Test loss - SEC: {test_loss_sec:7.4f} | '
            f'Test loss - SD: {test_loss_sd:7.4f} | '
            f'Test event accuracy: {test_event_accuracy:7.4f} | '
            f'Test speech accuracy: {test_speech_accuracy:7.4f} | '
            f'Test speech recall: {test_speech_recall:7.4f}')

    # Verification process on newly trained SD
    ver_targets, ver_predictions, ver_probs, ver_acc, ver_recall, ver_lowest_validation_loss, ver_acc_val, ver_recall_val = verification(
        feature_extractor, None, train_dataloader, val_dataloader, test_dataloader, ver_epochs, ver_patience, device, training, job_idx)

    # Verification process on newly trained GD
    gender_ver_targets, gender_ver_predictions, gender_ver_probs, gender_ver_acc, gender_ver_recall, \
        gender_ver_lowest_validation_loss, gender_ver_acc_val, gender_ver_recall_val = verification_gender(
        feature_extractor, None, train_dataloader, val_dataloader, test_dataloader, ver_epochs, ver_patience, device, training, job_idx
        )

    # Print verification results
    print(f'Verification accuracy - test data: {ver_acc:7.4f} | '
            f'Verification recall - test data: {ver_recall:7.4f} | '
            f'Verification acuuracy - val data: {ver_acc_val:7.4f} | '
            f'Verification recall - val data: {ver_recall_val:7.4f}')

    print(f'Gender verification accuracy - test data {gender_ver_acc:7.4f} | '
            f'Gender verification recall - test data {gender_ver_recall:7.4f}')

    # Return the numbers
    return test_loss_sec, test_loss_sd, test_event_accuracy, test_speech_accuracy, test_speech_recall, test_event_targets, test_event_predictions, \
            ver_acc, ver_recall, ver_lowest_validation_loss, ver_targets, ver_predictions, ver_probs, ver_acc_val, ver_recall_val, \
            gender_ver_acc, gender_ver_recall, gender_ver_lowest_validation_loss, gender_ver_targets, gender_ver_predictions, gender_ver_probs, gender_ver_acc_val, gender_ver_recall_val