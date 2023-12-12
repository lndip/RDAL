import numpy as np
import torch 
from torch import no_grad
from torch.optim import Adam
from pathlib import Path

from getting_and_init_the_data import get_all_data_loaders
from utils import MODELS_DIR


def forward_backward_pass_train(mask_unet:torch.nn.Module,
                          dataloader:torch.utils.data.DataLoader,
                          optimizer:torch.optim.Optimizer,
                          device:str) -> tuple:
    """Forward backward pass for training the pretrained masking network in "fixed mask" training

    :param mask_unet: The masking network for forward
    :type mask_unet: torch.nn.Module
    :param dataloader: The dataloader to forward
    :type dataloader: torch.utils.data.DataLoader
    :param optimizer: The optimizer to backward
    :type optimizer: torch.optim.Optimizer
    :param device: The current device 
    :type device: str
    :return: The model and the mean loss of the model over the batches
    :rtype: tuple
    """

    losses = []

    if optimizer is not None:
        mask_unet.train()
    else:
        mask_unet.eval()

    for batch in dataloader:
        if optimizer is not None:
            optimizer.zero_grad()

        # Get the batches
        speech_samples, speech_event_label, speech_speech_label, speech_event, speech_gender_label, \
        no_speech_samples, no_speech_event_label, no_speech_speech_label, no_speech_event, no_speech_gender_label = batch

        # Pass the data to the appropriate device.
        samples = speech_samples.float().to(device)
        event = speech_event.to(device)

        # Reshape the audio to (batch_size, 1, _, _)
        samples = torch.unsqueeze(samples, dim=1)
        event = torch.unsqueeze(event, dim=1)

        # Get the predictions of our model.
        mask = mask_unet(samples)
        samples_masked = torch.mul(samples, mask)

        # Calculate the loss of our model.
        loss = torch.nn.functional.l1_loss(input=samples_masked, target=event)

        # Back prppagation
        if optimizer is not None:
            loss.backward()

            optimizer.step()

        # Log the loss of the batch
        losses.append(loss.item())

    return mask_unet, np.mean(losses)


def train_masking_network(mask_unet:torch.nn.Module,
                        batch_size:int,
                        patience:int,
                        job_idx:int,
                        device:int,
                        training="fixed_mask",
                        epochs=5000) -> tuple:
    """Pretrain the source separation network

    :param mask_unet: The source separation network to pretrain
    :type mask_unet: torch.nn.Module
    :param batch_size: The batch size for training
    :type batch_size: int
    :param patience: The patience for training
    :type patience: int
    :param job_idx: The job index on the cluster
    :type job_idx: int
    :param device: The current device
    :type device: int
    :param training: The training mode, defaults to "fixed_mask"
    :type training: str, optional
    :param epochs: The numer of epochs for training, defaults to 5000
    :type epochs: int, optional
    :return: The training and validation losses over the epochs and the model
    :rtype: tuple
    """

    # Give the parameters of Unet to an optimizer.
    optimizer = Adam(params=mask_unet.parameters(), lr=1e-3)

    # Getting and initializing the data
    train_dataloader, val_dataloader, _ = get_all_data_loaders(batch_size)

    # Variables for the early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0

    # Epoch metrics
    epoch_train_loss = []
    epoch_val_loss = []

    # Start training
    for epoch in range(epochs):
        mask_unet, train_loss = forward_backward_pass_train(mask_unet, 
                                                train_dataloader, 
                                                optimizer=optimizer, 
                                                device=device)

        with no_grad():
            mask_unet, val_loss = forward_backward_pass_train(mask_unet, 
                                                        val_dataloader, 
                                                        optimizer=None, 
                                                        device=device)
        
        epoch_train_loss.append(train_loss)
        epoch_val_loss.append(val_loss)

        print(f'Epoch: {epoch:03d} | '
        f'Mean training loss: {train_loss:7.4f} | '
        f'Mean validation loss {val_loss:7.4f}')

        # Early stopping
        if val_loss < lowest_validation_loss:
            lowest_validation_loss = val_loss
            best_validation_epoch = epoch
            torch.save(mask_unet.state_dict(), Path(MODELS_DIR, training, f"{job_idx}", f"best_pretrained_mask.pt"))

        if epoch - best_validation_epoch > patience:
            print('\nExiting due to early stopping', end='\n\n')
            print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss}', end='\n\n')
            break

    # Load the best model into the mask
    mask_unet.load_state_dict(torch.load(Path(MODELS_DIR, training, f"{job_idx}", f"best_pretrained_mask.pt")))
    # Save the model's state dict
    print("Save fixed mask to ", Path(MODELS_DIR, training, f"{job_idx}", f"best_pretrained_mask.pt"))
    torch.save(mask_unet.state_dict(), Path(MODELS_DIR, training, f"{job_idx}", f"best_pretrained_mask.pt"))

    return epoch_train_loss, epoch_val_loss, mask_unet

