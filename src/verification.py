from utils import get_models, MODELS_DIR
from torch.utils.data import DataLoader

from torch.optim import SGD
from sklearn.metrics import accuracy_score, recall_score
from torchaudio.transforms import MelScale, AmplitudeToDB

import torch
import numpy as np
from pathlib import Path


def verification(feature_extractor:torch.nn.Module,
                mask_unet:torch.nn.Module,
                train_dataloader:DataLoader,
                val_dataloader:DataLoader,
                test_dataloader:DataLoader,
                epochs:int,
                patience:int,
                device:str,
                training:str,
                job_idx:int) -> tuple:
    """Train the verification speech classifier 

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
    :param epochs: The number of epochs in the training
    :type epochs: int
    :param patience: The patience of the training
    :type patience: int
    :param device: The current device
    :type device: str
    :param training: The specified training mode
    :type training: str
    :param job_idx: The job index on the cluster
    :type job_idx: int
    :return: The training and testing metrics
    :rtype: tuple
    """

    # Get the models
    _,_,_,_,verifying_sd, _ = get_models(device)

    # Get the optimizer
    optimizer_verify = SGD(verifying_sd.parameters(), lr=1e-2, momentum=0.9)


    # Variables for the early stopping
    lowest_validation_loss = 1.0e10
    best_epoch = 0

    for epoch in range(epochs):
        # Train the verifying_sd
        _,_,_,verifying_sd,train_loss,_,_ = forward_backward_verify(feature_extractor,
                                                                mask_unet,
                                                                verifying_sd,
                                                                train_dataloader,
                                                                optimizer_verify,
                                                                device)
        # Validate the verifying_sd
        with torch.no_grad():
            _,_,_,verifying_sd,val_loss,_,_ = forward_backward_verify(feature_extractor,
                                                        mask_unet,
                                                        verifying_sd,
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
            # best_verifying_sd = deepcopy(verifying_sd)
            torch.save(verifying_sd.state_dict(), Path(MODELS_DIR, training, f"{job_idx}", f"verifying_sd.pt"))
        elif epoch - best_epoch > patience:
            print(f'Early stopping at epoch {epoch:03d}, with a validation loss of {lowest_validation_loss:7.4f}')
            break

    # Test the verifying_sd
    verifying_sd.load_state_dict(torch.load(Path(MODELS_DIR, training, f"{job_idx}", f"verifying_sd.pt")))
    print(f"Loading SD model from {Path(MODELS_DIR, training, f'{job_idx}', 'verifying_sd.pt')}")
    test_targets, test_predictions, test_probs, _,_,test_acc, test_recall = forward_backward_verify(feature_extractor,
                                                                                                    mask_unet,
                                                                                                    verifying_sd,
                                                                                                    test_dataloader,
                                                                                                    None,
                                                                                                    device)
    _,_,_,_,_,val_acc,val_recall = forward_backward_verify(feature_extractor,
                                                        mask_unet,
                                                        verifying_sd,
                                                        val_dataloader,
                                                        None,
                                                        device)
    
    return test_targets, test_predictions, test_probs, test_acc, test_recall, lowest_validation_loss, val_acc, val_recall


def forward_backward_verify(feature_extractor:torch.nn.Module,
                            mask_unet:torch.nn.Module,
                            verifying_sd:torch.nn.Module,
                            dataloader:DataLoader,
                            optimizer_verify:torch.optim.Optimizer,
                            device:str) -> tuple:
    """Forward and backward pass for the verification speech classifier

    :param feature_extractor: The feature extractor
    :type feature_extractor: torch.nn.Module
    :param mask_unet: The source separation network if there is one
    :type mask_unet: torch.nn.Module
    :param verifying_sd: The speech classifier
    :type verifying_sd: torch.nn.Module
    :param dataloader: The dataloader
    :type dataloader: DataLoader
    :param optimizer_verify: The optimizer
    :type optimizer_verify: torch.optim.Optimizer
    :param device: The current device
    :type device: str
    :return: The updated model and metrics
    :rtype: tuple
    """

    verify_losses = []

    speech_targets = []
    speech_predictions = []

    melscale_transform = MelScale(n_mels=64, sample_rate=44100, n_stft=1411//2+1, norm='slaney', mel_scale='slaney').to(device)
    power_transform = AmplitudeToDB(stype="power", top_db=80).to(device)

    # Specify training or evalutation mode for models
    feature_extractor.eval()
    if mask_unet is not None:
        mask_unet.eval()
    if optimizer_verify is not None:
        verifying_sd.train()
    else:
        verifying_sd.eval()

    for batch in dataloader:
        if optimizer_verify is not None:
            optimizer_verify.zero_grad()
        
        # Get the batches
        speech_samples, speech_event_label, speech_speech_label, speech_event, speech_gender_label, \
        no_speech_samples, no_speech_event_label, no_speech_speech_label, no_speech_event, no_speech_gender_label = batch

        # Concatenate the speech and non speech samples into 1 tensor
        samples = torch.cat((speech_samples, no_speech_samples))
        speech_label = torch.cat((speech_speech_label, no_speech_speech_label))

        samples = samples.float().to(device).unsqueeze(dim=1)
        speech_label = speech_label.to(device)

        if mask_unet is not None:
            with torch.no_grad():
                # Get the masked audio
                mask = mask_unet(samples)
                samples_masked = torch.mul(mask,samples)
            # Calculate melband on masked spectrogram (batch_sz, 1, 64, 101)
            mel = melscale_transform(samples_masked**2)
        else:
            # Calculate melband on spectrogram (batch_sz, 1, 64, 101)
            mel = melscale_transform(samples**2)
        mel = power_transform(mel)

        with torch.no_grad():
            # Get the features
            features = feature_extractor(mel)

        # Get the predictions
        pred_speech_label = verifying_sd(features).squeeze(dim=1)

        # Calculate the loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(input=pred_speech_label, target=speech_label.float())
        
        # Back propagate
        if optimizer_verify is not None:
            loss.backward()
            optimizer_verify.step()

        # Save the loss
        verify_losses.append(loss.item())

        speech_targets.append(speech_label.cpu())
        speech_predictions.append(pred_speech_label.detach().cpu())

    # Concatenate the predictions and targets
    speech_targets = torch.cat(speech_targets)
    speech_predictions = torch.cat(speech_predictions)
    speech_probabilities = torch.sigmoid(speech_predictions)
    speech_predictions = (speech_probabilities >= 0.5).int()

    # Calculate the accuracy and recall
    sd_accuracy = accuracy_score(y_true=speech_targets.numpy(), y_pred=speech_predictions.numpy())
    sd_recall = recall_score(y_true=speech_targets.numpy(), y_pred=speech_predictions.numpy())

    return speech_targets.detach().cpu().numpy(), \
        speech_predictions.detach().cpu().numpy(), \
        speech_probabilities.detach().cpu().numpy(), \
        verifying_sd, \
        np.mean(verify_losses), \
        sd_accuracy, sd_recall