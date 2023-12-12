import torch
import numpy as np
from torch.nn.utils import clip_grad_value_
from sklearn.metrics import accuracy_score, recall_score
from torchaudio.transforms import MelScale, AmplitudeToDB


def forward_backward_pass_sec(feature_extractor: torch.nn.Module,
                            sound_event_classifier: torch.nn.Module,
                            data_loader: torch.utils.data.DataLoader,
                            optimizer: torch.optim.Optimizer,
                            device: str) -> tuple:
    """Forward and backward pass for the baseline

    :param feature_extractor: The feature extractor
    :type feature_extractor: torch.nn.Module
    :param sound_event_classifier: The sound event classifier
    :type sound_event_classifier: torch.nn.Module
    :param data_loader: The data loader
    :type data_loader: torch.utils.data.DataLoader
    :param optimizer: Training optimizer
    :type optimizer: torch.optim.Optimizer
    :param device: The current device
    :type device: str
    :return: The updated models, the losses, and the metrics
    :rtype: tuple
    """

    sec_losses = []

    event_targets = []
    event_predictions = []

    melscale_transform = MelScale(n_mels=64, sample_rate=44100, n_stft=1411//2+1, norm='slaney', mel_scale='slaney').to(device)
    power_transform = AmplitudeToDB(stype="power", top_db=80).to(device)

    if optimizer is not None:
        feature_extractor.train()
        sound_event_classifier.train()
    else:
        feature_extractor.eval()
        sound_event_classifier.eval()

    for batch in data_loader:
        # Zero the optimizer
        if optimizer is not None:
            optimizer.zero_grad()   
        
        # Get the batches
        speech_samples, speech_event_label, speech_speech_label, speech_event, speech_gender_label, \
        no_speech_samples, no_speech_event_label, no_speech_speech_label, no_speech_event, no_speech_gender_label = batch

        # Concatenate the speech and non speech samples into 1 tensor
        samples = torch.cat((speech_samples, no_speech_samples))
        event_label = torch.cat((speech_event_label, no_speech_event_label))

        samples = samples.float().to(device).unsqueeze(dim=1)
        event_label = event_label.to(device)

        mel = melscale_transform(samples**2)
        mel = power_transform(mel)

        # Get predictions
        features = feature_extractor(mel)
        pred_event_label = sound_event_classifier(features)

        # Loss calculation
        loss_sec = torch.nn.functional.cross_entropy(input=pred_event_label, target=event_label)

        if optimizer is not None:
            loss_sec.backward()
            optimizer.step()

        # Log the loss of the batch
        sec_losses.append(loss_sec.item())

        event_targets.append(event_label.cpu())
        event_predictions.append(pred_event_label.detach().cpu())

    # Record the predictions  
    event_targets = torch.cat(event_targets)
    event_predictions = torch.cat(event_predictions)

    # Calculate the metrics
    event_predictions = torch.argmax(torch.nn.functional.softmax(event_predictions, dim=1), dim=1)

    sec_accuracy = accuracy_score(event_targets, event_predictions)

    return feature_extractor, \
           sound_event_classifier,\
           np.mean(sec_losses), \
           sec_accuracy, \
           event_targets, \
           event_predictions


def forward_backward_pass_rdal(feature_extractor: torch.nn.Module,
                        sound_event_classifier: torch.nn.Module, 
                        speech_discriminator: torch.nn.Module,
                        data_loader: torch.utils.data.DataLoader, 
                        optimizer: torch.optim.Optimizer,
                        device: str,
                        do_grad_clip:bool,
                        grad_clip_value:float)-> tuple:
    """Forward and backward pass for RDAL

    :param feature_extractor: The feature extractor
    :type feature_extractor: torch.nn.Module
    :param sound_event_classifier: The sound event classifier
    :type sound_event_classifier: torch.nn.Module
    :param speech_discriminator: The speech discriminator
    :type speech_discriminator: torch.nn.Module
    :param data_loader: The data loader
    :type data_loader: torch.utils.data.DataLoader
    :param optimizer: Training optimizer
    :type optimizer: torch.optim.Optimizer
    :param device: The current device
    :type device: str
    :param do_grad_clip: Do gradient clipping?
    :type do_grad_clip: bool
    :param grad_clip_value: Gradient clipping value
    :type grad_clip_value: float
    :return: The updated models, the losses, and the metrics
    :rtype: tuple
    """

    sec_losses = []
    sd_losses = []

    event_targets = []
    event_predictions = []

    speech_targets = []
    speech_predictions = []

    melscale_transform = MelScale(n_mels=64, sample_rate=44100, n_stft=1411//2+1, norm='slaney', mel_scale='slaney').to(device)
    power_transform = AmplitudeToDB(stype="power", top_db=80).to(device)

    if optimizer is not None:
        # Training mode
        feature_extractor.train()
        sound_event_classifier.train()
        speech_discriminator.train()
    else:
        # Evaluation mode
        feature_extractor.eval()
        sound_event_classifier.eval()
        speech_discriminator.eval()
    
    for batch in data_loader:
        # Zero the optimizer
        if optimizer is not None:
            optimizer.zero_grad()

        # Get the batches
        speech_samples, speech_event_label, speech_speech_label, speech_event, speech_gender_label, \
        no_speech_samples, no_speech_event_label, no_speech_speech_label, no_speech_event, no_speech_gender_label = batch

        # Concatenate the speech and non speech samples into 1 tensor
        samples = torch.cat((speech_samples, no_speech_samples))
        event_label = torch.cat((speech_event_label, no_speech_event_label))
        speech_label = torch.cat((speech_speech_label, no_speech_speech_label))

        samples = samples.float().to(device).unsqueeze(dim=1)
        event_label = event_label.to(device)
        speech_label = speech_label.to(device)

        # Calculate melband on masked spectrogram (batch_sz, 1, h, w)
        mel = melscale_transform(samples**2)
        mel = power_transform(mel)

        # Get predictions
        features = feature_extractor(mel)
        pred_event_label = sound_event_classifier(features) # batch size, 3
        pred_speech_label = speech_discriminator(features).squeeze(dim=1)

        # Loss calculation
        loss_sec = torch.nn.functional.cross_entropy(input=pred_event_label, target=event_label)
        loss_sd = torch.nn.functional.binary_cross_entropy_with_logits(input=pred_speech_label, target=speech_label.float())
        
        if optimizer is not None:
            loss = loss_sec+loss_sd
            loss.backward()

            # Gradient clipping (if applicable)
            if do_grad_clip:
                clip_grad_value_(speech_discriminator.parameters(), grad_clip_value)
            
            optimizer.step()

        # Log the loss of the batch
        sec_losses.append(loss_sec.item())
        sd_losses.append(loss_sd.item())

        event_targets.append(event_label.cpu())
        event_predictions.append(pred_event_label.detach().cpu())

        speech_targets.append(speech_label.cpu())
        speech_predictions.append(pred_speech_label.detach().cpu())

    # Record the predictions
    event_targets = torch.cat(event_targets)
    event_predictions = torch.cat(event_predictions)

    speech_targets = torch.cat(speech_targets)
    speech_predictions = torch.cat(speech_predictions)

    # Calculate accuracy metrics
    event_predictions = torch.argmax(torch.nn.functional.softmax(event_predictions, dim=1), dim=1)
    speech_predictions = (torch.sigmoid(speech_predictions) >= 0.5).int()

    sec_accuracy = accuracy_score(y_true=event_targets.numpy(), y_pred=event_predictions.numpy())
    sd_accuracy = accuracy_score(y_true=speech_targets.numpy(), y_pred=speech_predictions.numpy())

    sd_recall = recall_score(y_true=speech_targets.numpy(), y_pred=speech_predictions.numpy())

    return feature_extractor, \
           sound_event_classifier, \
           speech_discriminator, \
           np.mean(sec_losses), \
           np.mean(sd_losses), \
           sec_accuracy, \
           sd_accuracy, \
           sd_recall,\
           event_targets, \
           event_predictions


def forward_backward_pass_rdal_mask(feature_extractor: torch.nn.Module,
                                    mask_unet: torch.nn.Module,
                                    sound_event_classifier: torch.nn.Module, 
                                    speech_discriminator: torch.nn.Module,
                                    data_loader: torch.utils.data.DataLoader, 
                                    optimizer: torch.optim.Optimizer,
                                    device: str,
                                    do_grad_clip:bool,
                                    grad_clip_value:float)-> tuple:
    """Forward and backward pass for RDAL+M

    :param feature_extractor: The feature extractor
    :type feature_extractor: torch.nn.Module
    :param mask_unet: The masking network
    :type mask_unet: torch.nn.Module
    :param sound_event_classifier: The sound event classifier
    :type sound_event_classifier: torch.nn.Module
    :param speech_discriminator: The speech discriminator
    :type speech_discriminator: torch.nn.Module
    :param data_loader: The data loader
    :type data_loader: torch.utils.data.DataLoader
    :param optimizer: The training optimizer
    :type optimizer: torch.optim.Optimizer
    :param device: The current device
    :type device: str
    :param do_grad_clip: Do gradient clipping?
    :type do_grad_clip: bool
    :param grad_clip_value: Gradient clipping value
    :type grad_clip_value: float
    :return: The updated models, the losses, and the metrics
    :rtype: tuple
    """

    sec_losses = []
    sd_losses = []

    event_targets = []
    event_predictions = []

    speech_targets = []
    speech_predictions = []

    melscale_transform = MelScale(n_mels=64, sample_rate=44100, n_stft=1411//2+1, norm='slaney', mel_scale='slaney').to(device)
    power_transform = AmplitudeToDB(stype="power", top_db=80).to(device)

    if optimizer is not None:
        # Training mode
        feature_extractor.train()
        sound_event_classifier.train()
        speech_discriminator.train()
        mask_unet.eval()
    else:
        # Evaluation mode
        feature_extractor.eval()
        mask_unet.eval()
        sound_event_classifier.eval()
        speech_discriminator.eval()
    
    for batch in data_loader:
        # Zero the optimizer
        if optimizer is not None:
            optimizer.zero_grad()

        # Get the batches
        speech_samples, speech_event_label, speech_speech_label, speech_event, speech_gender_label, \
        no_speech_samples, no_speech_event_label, no_speech_speech_label, no_speech_event, no_speech_gender_label = batch

        # Concatenate the speech and non speech samples into 1 tensor
        samples = torch.cat((speech_samples, no_speech_samples))
        event_label = torch.cat((speech_event_label, no_speech_event_label))
        speech_label = torch.cat((speech_speech_label, no_speech_speech_label))
        
        # Give the tensors to appropriate device
        samples = samples.float().to(device).unsqueeze(dim=1)
        event_label = event_label.to(device)
        speech_label = speech_label.to(device)

        # Get the masked audio
        mask = mask_unet(samples)
        samples_masked = torch.mul(mask,samples)

        # Calculate melband on masked spectrogram (batch_sz, 1, h, w)
        mel = melscale_transform(samples_masked**2)
        mel = power_transform(mel)

        # Get predictions
        features = feature_extractor(mel)
        pred_event_label = sound_event_classifier(features) # batch size, 3
        pred_speech_label = speech_discriminator(features).squeeze(dim=1)

        # Loss calculation
        loss_sec = torch.nn.functional.cross_entropy(input=pred_event_label, target=event_label)
        loss_sd = torch.nn.functional.binary_cross_entropy_with_logits(input=pred_speech_label, target=speech_label.float())
        
        if optimizer is not None:
            loss = loss_sec+loss_sd
            loss.backward()

            # Gradient clipping (if applicable)
            if do_grad_clip:
                clip_grad_value_(speech_discriminator.parameters(), grad_clip_value)
            
            optimizer.step()

        # Log the loss of the batch
        sec_losses.append(loss_sec.item())
        sd_losses.append(loss_sd.item())

        event_targets.append(event_label.cpu())
        event_predictions.append(pred_event_label.detach().cpu())

        speech_targets.append(speech_label.cpu())
        speech_predictions.append(pred_speech_label.detach().cpu())

    # Record the predictions
    event_targets = torch.cat(event_targets)
    event_predictions = torch.cat(event_predictions)

    speech_targets = torch.cat(speech_targets)
    speech_predictions = torch.cat(speech_predictions)

    # Calculate accuracy metrics
    event_predictions = torch.argmax(torch.nn.functional.softmax(event_predictions, dim=1), dim=1)
    speech_predictions = (torch.sigmoid(speech_predictions) >= 0.5).int()

    sec_accuracy = accuracy_score(y_true=event_targets.numpy(), y_pred=event_predictions.numpy())

    sd_accuracy = accuracy_score(y_true=speech_targets.numpy(), y_pred=speech_predictions.numpy())
    sd_recall = recall_score(y_true=speech_targets.numpy(), y_pred=speech_predictions.numpy())

    return feature_extractor, \
           mask_unet,\
           sound_event_classifier, \
           speech_discriminator, \
           np.mean(sec_losses), \
           np.mean(sd_losses), \
           sec_accuracy, \
           sd_accuracy, \
           sd_recall, \
           event_targets, \
           event_predictions