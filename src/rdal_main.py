import sys
import pickle
from pathlib import Path
import torch
from sklearn.metrics import roc_curve, roc_auc_score

from utils import PICKLE_RESULTS, MODELS_DIR, empty_dir
from rdal_train_test import train_rdal, test_rdal

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Process on {device}', '\n')

    if (str(device) == 'cuda'):
        print(f'Device name: {torch.cuda.get_device_name(0)}', '\n')

    # Get the job index
    job_idx = int(sys.argv[1])
    training = "rdal"

    model_dir = Path(MODELS_DIR, training, f"{job_idx}")
    model_dir.mkdir(parents=True, exist_ok=True)
    empty_dir(model_dir)

    # Set the parameters
    batch_size = 32
    epochs = 5000
    P=50

    print(f"Start training with {training} \n")

    # Train RDAL with mask
    epoch_train_loss_sec, epoch_train_loss_sd, epoch_val_loss_sec, epoch_val_loss_sd, \
    epoch_train_event_accuracy, epoch_train_speech_accuracy, epoch_train_speech_recall, \
    epoch_val_event_accuracy, epoch_val_speech_accuracy, epoch_val_speech_recall, \
    epoch_ver_accuracy, epoch_ver_recall, epoch_ver_lowest_validation_loss, \
    grl_parameters, best_epoch = train_rdal(batch_size=batch_size,
                                training=training,
                                do_grad_clip=True,
                                grad_clip_value=1,
                                P=P,
                                patience=10, ver_patience=50, job_idx=job_idx, epochs=epochs, device=device)

    # Test RDAL with mask
    print('\n\n')
    print(f"Start testing with {training}")
    test_loss_sec, test_loss_sd, test_event_accuracy, test_speech_accuracy, test_speech_recall, test_event_targets, test_event_predictions, \
    ver_acc, ver_recall, ver_lowest_validation_loss, ver_targets, ver_predictions, \
    ver_probabilities, ver_acc_val, ver_recall_val, \
    gender_ver_acc, gender_ver_recall, gender_ver_lowest_validation_loss, gender_ver_targets, \
    gender_ver_predictions, gender_ver_probs, gender_ver_acc_val, gender_ver_recall_val = test_rdal(batch_size=batch_size,
                                    training=training,
                                    job_idx=job_idx,
                                    ver_epochs=epochs,
                                    ver_patience=50, 
                                    device=device,
                                    best_epoch=best_epoch)

    # Calculate AUC score
    fpr, tpr, thresholds = roc_curve(ver_targets, ver_probabilities)
    auc_score = roc_auc_score(ver_targets, ver_probabilities)

    # Save the figures to a pickle file
    results_dict = {
        "P": P,
        "epoch_train_loss_sec": epoch_train_loss_sec,
        "epoch_train_loss_sd": epoch_train_loss_sd,
        "epoch_val_loss_sec": epoch_val_loss_sec,
        "epoch_val_loss_sd": epoch_val_loss_sd,
        "epoch_train_event_accuracy": epoch_train_event_accuracy,
        "epoch_train_speech_accuracy": epoch_train_speech_accuracy,
        "epoch_train_speech_recall": epoch_train_speech_recall,
        "epoch_val_event_accuracy": epoch_val_event_accuracy,
        "epoch_val_speech_accuracy": epoch_val_speech_accuracy,
        "epoch_val_speech_recall": epoch_val_speech_recall,
        "epoch_ver_accuracy": epoch_ver_accuracy,
        "epoch_ver_recall": epoch_ver_recall,
        "epoch_ver_lowest_validation_loss": epoch_ver_lowest_validation_loss,
        "grl_parameters": grl_parameters,
        "test_loss_sec": test_loss_sec,
        "test_loss_sd": test_loss_sd,
        "test_event_accuracy": test_event_accuracy,
        "test_speech_accuracy": test_speech_accuracy,
        "test_speech_recall": test_speech_recall,
        "test_event_targets": test_event_targets,
        "test_event_predictions": test_event_predictions,
        "ver_acc": ver_acc,
        "ver_recall": ver_recall,
        "ver_lowest_validation_loss": ver_lowest_validation_loss,
        "ver_targets": ver_targets,
        "ver_predictions": ver_predictions,
        "ver_probabilities": ver_probabilities,
        "ver_acc_val": ver_acc_val,
        "ver_recall_val": ver_recall_val,
        "auc_score": auc_score,
        "gender_ver_acc": gender_ver_acc,
        "gender_ver_recall": gender_ver_recall,
        "gender_ver_lowest_validation_loss" : gender_ver_lowest_validation_loss,
        "gender_ver_targets" : gender_ver_targets,
        "gender_ver_predictions" : gender_ver_predictions,
        "gender_ver_probs": gender_ver_probs,
        "gender_ver_acc_val" : gender_ver_acc_val,
        "gender_ver_recall_val" : gender_ver_recall_val
    }

    with open(Path(PICKLE_RESULTS, training, f"rdal_{training}_{job_idx}.pickle"), "wb") as f:
        print(f"Saving results to {Path(PICKLE_RESULTS, training, f'rdal_{training}_{job_idx}.pickle')}")
        pickle.dump(results_dict, f)

if __name__ == "__main__":
    main()