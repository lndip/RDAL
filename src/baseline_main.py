import sys
import pickle
from pathlib import Path
import torch
from sklearn.metrics import roc_auc_score

from utils import PICKLE_RESULTS, MODELS_DIR, empty_dir
from baseline_train_test import train_sec_baseline, test_sec_sd_baseline

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Process on {device}', '\n\n')

    if (str(device) == 'cuda'):
        print(f'Device name: {torch.cuda.get_device_name(0)}', '\n')

    # Get the job index and set the seed
    job_idx = int(sys.argv[1])

    # Set the parameters
    batch_size = 32
    epochs = 5000

    training = "baseline"
    model_dir = Path(MODELS_DIR, training, f"{job_idx}")
    model_dir.mkdir(parents=True, exist_ok=True)
    empty_dir(model_dir)

    print(f"Start training with {training}")

    # Train SEC
    epoch_train_loss_sec, epoch_val_loss_sec, epoch_train_event_accuracy, epoch_val_event_accuracy = train_sec_baseline(batch_size, 
                                                                                                            patience=50, 
                                                                                                            job_idx=job_idx,
                                                                                                            training=training,
                                                                                                            epochs=epochs, 
                                                                                                            device=device)


    # Test SEC SD
    print('\n\n')
    print(f"Start testing with {training}")

    test_loss_sec, test_event_accuracy, test_event_targets, test_event_predictions, sd_acc, sd_recall, sd_targets, sd_predictions, sd_probs, sd_acc_val, sd_acc_recall, \
        gender_ver_acc, gender_ver_recall, gender_ver_lowest_validation_loss, gender_ver_targets, gender_ver_predictions, \
        gender_ver_probs, gender_ver_acc_val, gender_ver_recall_val = test_sec_sd_baseline(batch_size, 
                                                                                training=training, 
                                                                                job_idx=job_idx, 
                                                                                device=device, 
                                                                                sd_epochs=epochs, 
                                                                                sd_patience=50)

    # Plot ROC curve and save plot
    auc_score = roc_auc_score(sd_targets, sd_probs)

    # Save the results
    results_dict = {
        "epoch_train_loss_sec": epoch_train_loss_sec,
        "epoch_val_loss_sec": epoch_val_loss_sec,
        "epoch_train_event_accuracy": epoch_train_event_accuracy,
        "epoch_val_event_accuracy": epoch_val_event_accuracy,
        "test_loss_sec": test_loss_sec,
        "test_event_accuracy": test_event_accuracy,
        "test_event_targets": test_event_targets,
        "test_event_predictions": test_event_predictions,
        "sd_acc": sd_acc,
        "sd_recall": sd_recall,
        "sd_targets": sd_targets,
        "sd_predictions": sd_predictions,
        "sd_probs": sd_probs,
        "sd_acc_val": sd_acc_val,
        "sd_recall_val": sd_acc_recall,
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

    with open(Path(PICKLE_RESULTS, training, f"rdal_mask_{training}_{job_idx}.pickle"), "wb") as f:
        print(f"Saving results to {Path(PICKLE_RESULTS, training, f'rdal_mask_{training}_{job_idx}.pickle')}")
        pickle.dump(results_dict, f)


if __name__ == "__main__":
    main()   
