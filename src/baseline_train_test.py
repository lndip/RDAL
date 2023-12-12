from forward_backward_pass import forward_backward_pass_sec
from getting_and_init_the_data import get_all_data_loaders
from utils import (get_models, 
                   MODELS_DIR)
from verification import verification
from verification_gender import verification_gender

import torch

from torch.optim import Adam
from pathlib import Path

def train_sec_baseline(batch_size:int,
              patience:int,
              job_idx:int,
              training:str,
              device:str,
              epochs=5000) -> tuple:
    """Train the baseline

    :param batch_size: Batch size 
    :type batch_size: int
    :param patience: Patience for the early stopping
    :type patience: int
    :param job_idx: The job index on the cluster
    :type job_idx: int
    :param training: The specified training mode
    :type training: str
    :param device: The current device
    :type device: str
    :param epochs: Number of training epochs, defaults to 5000
    :type epochs: int, optional
    :return: The training and validation metrics
    :rtype: tuple
    """

    # Get the models
    feature_extractor, _, sound_event_classifier, _, _,_ = get_models(device)

    # Get the data loaders
    train_dataloader, val_dataloader, test_dataloader = get_all_data_loaders(batch_size)

    # Get the optimizer
    optimizer =  Adam(list(feature_extractor.parameters()) + list(sound_event_classifier.parameters()), lr=1e-3)

    # Variables for the early stopping
    best_epoch = 0
    lowest_validation_loss = 1.0e10

    # Epochs metrics
    epoch_train_loss_sec = []
    epoch_val_loss_sec = []

    epoch_train_event_accuracy = []
    epoch_val_event_accuracy = []

    for epoch in range(epochs):
        feature_extractor, sound_event_classifier, train_loss_sec, train_event_accuracy,_,_ = \
            forward_backward_pass_sec(feature_extractor,
                                    sound_event_classifier,
                                    train_dataloader,
                                    optimizer,
                                    device)

        with torch.no_grad():
            feature_extractor, sound_event_classifier, val_loss_sec, val_event_accuracy, _,_ = \
                forward_backward_pass_sec(feature_extractor,
                                        sound_event_classifier,
                                        val_dataloader,
                                        None,
                                        device)

        # Append the metrics
        epoch_train_loss_sec.append(train_loss_sec)
        epoch_val_loss_sec.append(val_loss_sec)
        epoch_train_event_accuracy.append(train_event_accuracy)
        epoch_val_event_accuracy.append(val_event_accuracy)

        # Print the metrics
        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss - SEC: {epoch_train_loss_sec[-1]:7.4f} | '
              f'Mean validation loss - SEC: {epoch_val_loss_sec[-1]:7.4f} ')

        # Early stopping
        if val_loss_sec < lowest_validation_loss:
            lowest_validation_loss = val_loss_sec
            best_epoch = epoch
            torch.save(feature_extractor.state_dict(), Path(MODELS_DIR, training, f"{job_idx}", f"feature_extractor.pt"))
            torch.save(sound_event_classifier.state_dict(), Path(MODELS_DIR, training, f"{job_idx}", f"sound_event_classifier.pt"))

        if epoch - best_epoch > patience:
            print(f'Early stopping at epoch {epoch}, best epoch was {best_epoch}')
            break

    return epoch_train_loss_sec, epoch_val_loss_sec, epoch_train_event_accuracy, epoch_val_event_accuracy


def test_sec_sd_baseline(batch_size:int, 
                training:str, 
                job_idx:int, 
                device:str, 
                sd_epochs:int, 
                sd_patience:int) -> tuple:
    """Test the baseline

    :param batch_size: Batch size
    :type batch_size: int
    :param training: The specified training mode
    :type training: str
    :param job_idx: The job index on the cluster
    :type job_idx: int
    :param device: The current device
    :type device: str
    :param sd_epochs: The number of epochs in the training speech discriminator
    :type sd_epochs: int
    :param sd_patience: The patience of the training speech discriminator
    :type sd_patience: int
    :return: The testing metrics
    :rtype: tuple
    """

    # Get the models
    feature_extractor, _, sound_event_classifier, _, _,_ = get_models(device)

    # Get the data loaders
    train_dataloader, val_dataloader, test_dataloader = get_all_data_loaders(batch_size)

    # Load the best models
    feature_extractor.load_state_dict(torch.load(Path(MODELS_DIR, training, f"{job_idx}", f"feature_extractor.pt"), map_location=device))
    sound_event_classifier.load_state_dict(torch.load(Path(MODELS_DIR, training, f"{job_idx}", f"sound_event_classifier.pt"), map_location=device))

    with torch.no_grad():
        feature_extractor, sound_event_classifier, test_loss_sec, test_event_accuracy, test_event_targets, test_event_preditions = \
            forward_backward_pass_sec(feature_extractor,
                                    sound_event_classifier,
                                    test_dataloader,
                                    None,
                                    device)

    print(f'Test loss - SEC: {test_loss_sec:7.4f} | '
        f'Test event accuracy: {test_event_accuracy:7.4f}')

    print("Start training SD")
    sd_targets, sd_predictions, sd_probs, sd_acc, sd_recall, sd_lowest_validation_loss, sd_acc_val, sd_acc_recall = verification(
        feature_extractor, None, train_dataloader, val_dataloader, test_dataloader, sd_epochs, sd_patience, device, training, job_idx)
    
    # Verification process on newly trained GD
    gender_ver_targets, gender_ver_predictions, gender_ver_probs, gender_ver_acc, gender_ver_recall, \
        gender_ver_lowest_validation_loss, gender_ver_acc_val, gender_ver_recall_val = verification_gender(
        feature_extractor, None, train_dataloader, val_dataloader, test_dataloader, sd_epochs, sd_patience, device, training, job_idx)

    # Print SD results
    print(f'Test speech accuracy: {sd_acc:7.4f} | '
        f'Test speech recall: {sd_recall:7.4f} | '
        f'Val speech accuracy: {sd_acc_val:7.4f} | '
        f'Val speech recall: {sd_acc_recall:7.4f}')
    
    print(f'Gender verification accuracy - test data {gender_ver_acc:7.4f} | '
            f'Gender verification recall - test data {gender_ver_recall:7.4f}')

    return test_loss_sec, test_event_accuracy, test_event_targets, test_event_preditions,\
            sd_acc, sd_recall, sd_targets, sd_predictions, sd_probs, sd_acc_val, sd_acc_recall, \
            gender_ver_acc, gender_ver_recall, gender_ver_lowest_validation_loss, gender_ver_targets, gender_ver_predictions, gender_ver_probs, gender_ver_acc_val, gender_ver_recall_val