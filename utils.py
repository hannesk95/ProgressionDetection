import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch

def seed_everything(seed: int):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def track_training_progress(train_loss_list, val_loss_list,
                            train_auc_list, val_auc_list,
                            train_bacc_list, val_bacc_list,
                            train_mcc_list, val_mcc_list,
                            train_f1_list, val_f1_list,
                            lr_list, result_dir: str = "./results/"):

    plt.figure(figsize=(25, 25))
    plt.subplot(6, 1, 1)
    plt.plot(np.arange(len(train_loss_list)), train_loss_list, label="train")
    plt.plot(np.arange(len(val_loss_list)), val_loss_list, label="val")
    plt.title("Loss")
    plt.grid()
    plt.legend()

    plt.subplot(6, 1, 2)
    plt.plot(np.arange(len(train_auc_list)), train_auc_list, label="train")
    plt.plot(np.arange(len(val_auc_list)), val_auc_list, label="val")
    plt.title("AUROC")
    plt.grid()
    plt.legend()

    plt.subplot(6, 1, 3)
    plt.plot(np.arange(len(train_bacc_list)), train_bacc_list, label="train")
    plt.plot(np.arange(len(val_bacc_list)), val_bacc_list, label="val")
    plt.title("Balanced Accuracy")
    plt.grid()
    plt.legend()

    plt.subplot(6, 1, 4)
    plt.plot(np.arange(len(train_mcc_list)), train_mcc_list, label="train")
    plt.plot(np.arange(len(val_mcc_list)), val_mcc_list, label="val")
    plt.title("MCC")
    plt.grid()
    plt.legend()

    plt.subplot(6, 1, 5)
    plt.plot(np.arange(len(train_f1_list)), train_f1_list, label="train")
    plt.plot(np.arange(len(val_f1_list)), val_f1_list, label="val")
    plt.title("F1-Score")
    plt.grid()
    plt.legend()

    plt.subplot(6, 1, 6)
    plt.plot(np.arange(len(lr_list)), lr_list)
    plt.title("Learning Rate")
    plt.grid()

    plt.savefig(os.path.join(result_dir, "performance.png"))
    plt.close()

