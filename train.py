import os
import logging
import warnings
import random
import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import profiler
import matplotlib.pyplot as plt
from time import time
from datetime import datetime

from model import ViT
from utils import save_checkpoint, set_logger, lastest_checkpoint
from data_processing import prepare_data, prepare_test_data


class Trainer:

    def __init__(self, model, optimizer, loss_func: str, device: str, scheduler):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = device
        self.scheduler = scheduler

    def train(self, trainloader, valloader, epochs: int, early_stop_patience: int = 5):

        train_losses, val_losses, accuracies = [], [], []
        best_val_loss = float('inf')
        patience_counter = 0

        try:
            outdir = os.path.join("Logs", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

            for epoch in range(epochs):
                start = time()
                train_loss = self.train_epoch(trainloader)
                val_accuracy, val_loss = self.test(valloader)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                accuracies.append(val_accuracy)

                logging.info(
                    f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.4f}, Time: {time() - start:.4f}")

                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_checkpoint(self.model, epoch + 1, outdir)
                    logging.info("-------- Saved Best Model! --------")
                else:
                    patience_counter += 1
                    logging.info("Early Stop Left: {}".format(early_stop_patience - patience_counter))

                if (early_stop_patience - patience_counter) == 0:
                    logging.info("-------- Early Stop! --------")
                    break

            save_checkpoint(self.model, epochs, outdir)

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt detected. Saving the model...")
            save_checkpoint(self.model, epoch + 1, outdir)
            logging.info("Model saved successfully.")

        return train_losses, val_losses, accuracies

    def train_epoch(self, trainloader):

        self.model.train()
        total_loss = 0

        for batch in trainloader:
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_func(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(images)
            self.scheduler.step(loss)

        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def test(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                logits = self.model(images)

                loss = self.loss_func(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def plot_metrics(train_losses, val_losses, accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join("Logs", "metrics.png"))
    plt.show()


@torch.no_grad()
def test(model, device, testloader, loss_func):
    model.eval()
    total_loss = 0
    correct = 0
    for batch in testloader:
        batch = [t.to(device) for t in batch]
        images, labels = batch

        logits = model(images)
        loss = loss_func(logits, labels)

        predictions = torch.argmax(logits, dim=1)
        correct += torch.sum(predictions == labels).item()
        total_loss += loss.item()

    accuracy = correct / len(testloader.dataset)
    avg_loss = total_loss / len(testloader.dataset)
    return accuracy, avg_loss


def test_visualize(model, device, testloader, classes):
    model.eval()
    with torch.no_grad():
        for batch in testloader:
            batch = [t.to(device) for t in batch]
            images, labels = batch

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            for i in range(len(images)):
                image = images[i]
                label = labels[i]
                prediction = predictions[i]

                plt.imshow(image.permute(1, 2, 0).cpu())
                plt.title(f"Label: {classes[label]}, Prediction: {classes[prediction]}")
                plt.show()


def setup_seed(seed=3407):
    os.environ['VIT'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministics = True
    torch.backends.cudnn.benchmarks = False
    torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "data/resized"
log_path = os.path.join("Logs", "train_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".log")
num_workers = 0
batch_size = 1280
epochs = 80
learning_rate = 1e-4
patch_size = 16
hidden_size = 64
num_hidden_layers = 3
num_attention_heads = 5
intermediate_size = 4 * hidden_size
hidden_dropout_prob = 0.092351408873614
attention_probs_dropout_prob = 0.02387247516356817
image_size = 224
num_channels = 3
qkv_bias = True
early_stop_patience = 5

def main(continue_train: bool = False, testing: bool = False, test_data_dir: str = "data/test"):
    if testing is False:
        set_logger(log_path)

        logging.info("-------- Start Building Dataset! --------")
        trainloader, valloader, testloader, classes = prepare_data(data_dir, batch_size=batch_size,
                                                                   num_workers=num_workers)
        logging.info("-------- Dataset Build! --------\n\n")

        logging.info("-------- Start Building Model! --------")
        model = ViT(image_size, hidden_size, num_hidden_layers, intermediate_size, len(classes), num_attention_heads,
                    hidden_dropout_prob, attention_probs_dropout_prob, num_channels, patch_size, qkv_bias).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        loss_func = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, loss_func, device, scheduler)
        logging.info("-------- Model Build! --------\n\n")

        if continue_train is False:
            train_losses, val_losses, accuracies = trainer.train(trainloader, valloader, epochs, early_stop_patience)
        else:
            model_path = lastest_checkpoint()

            # Check if a checkpoint exists before attempting to load it
            if model_path and os.path.isfile(model_path):
                model.load_state_dict(torch.load(model_path))
                logging.info(f"-------- Loaded Model from {model_path}! --------")
            else:
                logging.info("No valid checkpoint found. Starting training from scratch.")

            train_losses, val_losses, accuracies = trainer.train(trainloader, valloader, epochs, early_stop_patience)

        logging.info("-------- Training Finished! --------\n\n")

        logging.info("-------- Start Testing! --------")
        accuracy, avg_loss = trainer.test(testloader)
        logging.info(f"Test loss: {avg_loss:.4f}, Test accuracy: {accuracy:.4f}")
        logging.info("-------- Testing Finished! --------\n\n")

        plot_metrics(train_losses, val_losses, accuracies)

    else:
        loss_func = nn.CrossEntropyLoss()
        model_path = lastest_checkpoint()

        # Check if a checkpoint exists before loading
        if model_path and os.path.isfile(model_path):
            testloader, classes = prepare_test_data(test_data_dir, batch_size=batch_size, num_workers=num_workers)
            model = ViT(image_size, hidden_size, num_hidden_layers, intermediate_size, len(classes),
                        num_attention_heads,
                        hidden_dropout_prob, attention_probs_dropout_prob, num_channels, patch_size, qkv_bias)
            model.load_state_dict(torch.load(model_path))
            accuracy, avg_loss = test(model, device, testloader, loss_func)
            print(f"Test loss: {avg_loss:.4f}, Test accuracy: {accuracy:.4f}")
        else:
            logging.error("No checkpoint found. Cannot proceed with testing.")


def objective(trial):
    epochs = trial.suggest_int('epochs', 80, 250)
    learning_rate = trial.suggest_loguniform("learning_rate", 0.001, 0.01)
    patch_size = trial.suggest_categorical("patch_size", [16, 32])
    hidden_size = trial.suggest_categorical("hidden_size", [32, 48, 64])
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 2, 5)
    num_attention_heads = trial.suggest_int("num_attention_heads", 2, 5)
    hidden_dropout_prob = trial.suggest_uniform("hidden_dropout_prob", 0.0, 0.5)
    attention_probs_dropout_prob = trial.suggest_uniform("attention_probs_dropout_prob", 0.0, 0.5)

    intermediate_size = 4 * hidden_size

    logging.info("-------- Start Building Dataset! --------")
    trainloader, valloader, testloader, classes = prepare_data(data_dir, batch_size=batch_size, num_workers=num_workers)
    logging.info("-------- Dataset Build! --------\n\n")

    model = ViT(image_size, hidden_size, num_hidden_layers, intermediate_size, len(classes), num_attention_heads,
                hidden_dropout_prob,
                attention_probs_dropout_prob, num_channels, patch_size, qkv_bias).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    loss_func = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_func, device, scheduler)

    for epoch in range(epochs):
        start = time()
        train_loss = trainer.train_epoch(trainloader)
        end = time()
        val_accuracy, val_loss = trainer.test(valloader)

        logging.info(
            f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.4f}, Time: {end - start:.4f}")

    accuracy, val_loss = trainer.test(testloader)
    logging.info(f"Test loss: {val_loss:.4f}, Test accuracy: {accuracy:.4f}")

    return val_loss


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    setup_seed(42)
    # Profiling for time analysis
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
    #     main(continue_train=False, testing=False)
    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    main(continue_train=False, testing=False, test_data_dir="data/test")

# SMBO Hyperparameter Searching
    set_logger(os.path.join("Logs", "best_param_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".log"))
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    logging.info('Best trial:')
    trial = study.best_trial

    logging.info('Value: {}'.format(trial.value))
    logging.info('Params: ')
    for key, value in trial.params.items():
        logging.info('{}: {}'.format(key, value))