
import os
import time
import random
import logging
import numpy as np
import torch
import torchvision

from datetime import datetime

import copy
from tqdm import tqdm

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_timestamp():
    return datetime.now().strftime('%m%d-%H')

def setup_logger(logger_name, expt, root, level=logging.INFO, screen=True):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, expt + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

class CIFAR10(torch.utils.data.Dataset):

    def __init__(self, args, train=True, transform=None):
        self.data = torchvision.datasets.CIFAR10(
            root=args.data_path, train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    

def evaluate_model(args, task_model, loader, criterion):

    task_model.eval()
    task_model.to(args.device)

    running_loss = 0.0
    running_corrects = 0

    total_labels = 0
    since = time.time()
    for data in loader:
        inputs, labels = data
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        
        with torch.no_grad():
            logits = task_model(inputs)

        loss = criterion(logits, labels).item()

        _, preds = torch.max(logits.data, 1)

        running_loss += loss * inputs.size(0)

        total_labels += labels.size(0)

        running_corrects += torch.sum(preds == labels.data).item()
    time_elapsed = time.time() - since
    eval_loss = running_loss / total_labels
    eval_accuracy = running_corrects / total_labels
    
    return eval_loss, eval_accuracy, time_elapsed


def train_models(args, task_model, train_loader, val_loader, criterion, optimizer, scheduler, logger, wandb):

    task_model.to(args.device)   

    since = time.time()

    best_val_acc = 0.0
    best_train_acc = 0.0
    best_wt = copy.deepcopy(task_model.state_dict())

    for epoch in range(args.epochs):

        running_ce_loss = 0.0
        running_corrects = 0

        task_model.train()

        for data in tqdm(train_loader):

            inputs, labels = data

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)


            optimizer.zero_grad()

            logits = task_model(inputs)
           
            _, preds = torch.max(logits, 1)

            task_loss = criterion(logits, labels)

            task_loss.backward()
            optimizer.step()

            running_ce_loss += task_loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()


        scheduler.step()
        
        
        train_ce_loss = running_ce_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)


        val_loss, val_accuracy, _ = evaluate_model(
            args=args,
            task_model=task_model,
            loader=val_loader,
            criterion=criterion
        )

        log_msg = {
            "Epoch": epoch + 1,
            "Train CE Loss": train_ce_loss,
            "Train Accuracy": train_accuracy * 100,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy * 100
        }

        logger.info(log_msg)
        if wandb:
            wandb.log(log_msg)
        

        if val_accuracy > best_val_acc:

            best_val_acc = val_accuracy
            best_train_acc = train_accuracy
            best_wt = copy.deepcopy(task_model.state_dict())

    time_elapsed = time.time() - since
    final_log = {
        "Training Time": time_elapsed,
        "Best Validation Accuracy": best_val_acc * 100,
        "Best Training Accuracy": best_train_acc * 100
    }
    task_model.load_state_dict(best_wt)
        
    return final_log, task_model