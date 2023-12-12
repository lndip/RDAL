from utils import MODELS_DIR
from verifying_gd import VerifyingGD
from parameters import output_feature_length
from torch.utils.data import DataLoader

from torch.optim import SGD
from sklearn.metrics import accuracy_score, recall_score
from torchaudio.transforms import MelScale, AmplitudeToDB

import torch
import numpy as np
from pathlib import Path


def verification_gender(feature_extractor:torch.nn.Module,
                mask_unet:torch.nn.Module,
                train_dataloader:DataLoader,
                val_dataloader:DataLoader,
                test_dataloader:DataLoader,
                epochs:int,
                patience:int,
                device:str,
                training:str,
                job_idx:int) -> tuple:
    """Train the verification gender discriminator

    :param feature_extractor: The feature extractor
    :type feature_extractor: torch.nn.Module
    :param mask_unet: The source separation network if there is one
    :type mask_unet: torch.nn.Module
    :param train_dataloader: The train dataloader
    :type train_dataloader: DataLoader
    :param val_dataloader: The validation dataloader
    :type val_dataloader: DataLoader
    :param test_dataloader: The test dataloader
    :type test_dataloader: DataLoader
    :param epochs: Number of epochs in the training
    :type epochs: int
    :param patience: Patience of the training
    :type patience: int
    :param device: The current device
    :type device: str
    :param training: The specified training mode
    :type training: str
    :param job_idx: The job index on the cluster
    :type job_idx: int
    :return: The testing metrics
    :rtype: tuple
    """

    # Get the models
    verifying_gd = VerifyingGD(features_length=output_feature_length).to(device)

    # Get the optimizer
    optimizer_verify = SGD(verifying_gd.parameters(), lr=1e-2, momentum=0.9)

    # Variables for the early stopping
    lowest_validation_loss = 1.0e10
    best_epoch = 0

    for epoch in range(epochs):
        # Train the verifying_gd
        _,_,_,verifying_gd,train_loss,_,_ = forward_backward_verify(feature_extractor,
                                                                mask_unet,
                                                                verifying_gd,
                                                                train_dataloader,
                                                                optimizer_verify,
                                                                device)
        # Validate the verifying_gd
        with torch.no_grad():
            _,_,_,verifying_gd,val_loss,_,_ = forward_backward_verify(feature_extractor,
                                                        mask_unet,
                                                        verifying_gd,
                                                        val_dataloader,
                                                        None,
                                                        device)
        
        print(f'Epoch: {epoch:03d} | '
        f'Mean training loss: {train_loss:7.4f} | '
        f'Mean validation loss {val_loss:7.4f}')

        # Early stopping
        if val_loss < lowest_validation_loss:
            lowest_validation_loss = val_loss
            best_epoch = epoch
            torch.save(verifying_gd.state_dict(), Path(MODELS_DIR, training, f"{job_idx}", f"verifying_gd.pt"))
        elif epoch - best_epoch > patience:
            print(f'Early stopping at epoch {epoch:03d}, with a validation loss of {lowest_validation_loss:7.4f}')
            break

    # Test the verifying_gd
    verifying_gd.load_state_dict(torch.load(Path(MODELS_DIR, training, f"{job_idx}", f"verifying_gd.pt")))
    print(f"Loading GD model from {Path(MODELS_DIR, training, f'{job_idx}', 'verifying_gd.pt')}")
    test_targets, test_predictions, test_probs, _,_,test_acc, test_recall = forward_backward_verify(feature_extractor,
                                                                                                    mask_unet,
                                                                                                    verifying_gd,
                                                                                                    test_dataloader,
                                                                                                    None,
                                                                                                    device)
    _,_,_,_,_,val_acc,val_recall = forward_backward_verify(feature_extractor,
                                                        mask_unet,
                                                        verifying_gd,
                                                        val_dataloader,
                                                        None,
                                                        device)
    
    return test_targets, test_predictions, test_probs, test_acc, test_recall, lowest_validation_loss, val_acc, val_recall


def forward_backward_verify(feature_extractor:torch.nn.Module,
                            mask_unet:torch.nn.Module,
                            verifying_gd:torch.nn.Module,
                            dataloader:DataLoader,
                            optimizer_verify:torch.optim.Optimizer,
                            device:str) -> tuple:
    """Forward backward of the verification process

    :param feature_extractor: The feature extractor
    :type feature_extractor: torch.nn.Module
    :param mask_unet: The source separation network if there is one
    :type mask_unet: torch.nn.Module
    :param verifying_gd: The gender discriminator for verification
    :type verifying_gd: torch.nn.Module
    :param dataloader: The dataloader
    :type dataloader: DataLoader
    :param optimizer_verify: The optimizer for the verification
    :type optimizer_verify: torch.optim.Optimizer
    :param device: The current device
    :type device: str
    :return: The updated models and the metrics
    :rtype: tuple
    """

    verify_losses = []

    gender_targets = []
    gender_predictions = []

    melscale_transform = MelScale(n_mels=64, sample_rate=44100, n_stft=1411//2+1, norm='slaney', mel_scale='slaney').to(device)
    power_transform = AmplitudeToDB(stype="power", top_db=80).to(device)

    # Specify training or evalutation mode for models
    feature_extractor.eval()
    if mask_unet is not None:
        mask_unet.eval()
    if optimizer_verify is not None:
        verifying_gd.train()
    else:
        verifying_gd.eval()

    for batch in dataloader:
        if optimizer_verify is not None:
            optimizer_verify.zero_grad()

        # Get the batches
        speech_samples, speech_event_label, speech_speech_label, speech_event, speech_gender_label, \
        no_speech_samples, no_speech_event_label, no_speech_speech_label, no_speech_event, no_speech_gender_label = batch
        
        samples = speech_samples.float().to(device).unsqueeze(dim=1)
        gender_label = speech_gender_label.to(device)

        if mask_unet is not None:
            with torch.no_grad():
                # Get the masked audio
                mask = mask_unet(samples)
                samples_masked = torch.mul(mask,samples)

            # Calculate melband on masked spectrogram (batch_sz, 1, 64, 101)
            mel = melscale_transform(samples_masked**2)
        else:
            mel = melscale_transform(samples**2)
        mel = power_transform(mel)

        with torch.no_grad():
            # Get the features
            features = feature_extractor(mel)

        # Get the predictions
        pred_gender_label = verifying_gd(features).squeeze(dim=1)

        # Calculate the loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(input=pred_gender_label, target=gender_label.float())
        
        # Back propagate
        if optimizer_verify is not None:
            loss.backward()
            optimizer_verify.step()

        # Save the loss
        verify_losses.append(loss.item())

        gender_targets.append(gender_label.cpu())
        gender_predictions.append(pred_gender_label.detach().cpu())

    # Concatenate the predictions and targets
    gender_targets = torch.cat(gender_targets)
    gender_predictions = torch.cat(gender_predictions)
    gender_probabilities = torch.sigmoid(gender_predictions)
    gender_predictions = (gender_probabilities >= 0.5).int()

    # Calculate the accuracy and recall
    gd_accuracy = accuracy_score(y_true=gender_targets.numpy(), y_pred=gender_predictions.numpy())
    gd_recall = recall_score(y_true=gender_targets.numpy(), y_pred=gender_predictions.numpy())

    return gender_targets.detach().cpu().numpy(), \
        gender_predictions.detach().cpu().numpy(), \
        gender_probabilities.detach().cpu().numpy(), \
        verifying_gd, \
        np.mean(verify_losses), \
        gd_accuracy, gd_recall