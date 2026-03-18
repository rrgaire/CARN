
import os
import time
import random
import logging
import numpy as np
import torch
import thop
import torchvision
import torch.nn.functional as F

from datetime import datetime

from fvcore.nn import FlopCountAnalysis

import copy
from tqdm import tqdm
import torch.nn as nn

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


class CIFAR10WithSampler(torch.utils.data.Dataset):
    def __init__(self, args, model, transform, train=True):
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=args.data_path,
            download=True,
            train=train,
            transform=transform
        )
        
        self.device = args.device
        self.model = model
        self.alpha = args.alpha
        self.beta = args.beta
        
        self.energy_costs = self.compute_energy_costs()
        self.entropies, self.confidences, self.complexity_scores, self.complexity_labels = self.compute_sample_metrics()
        
        self.weights = torch.tensor(
            [len(self.complexity_labels) / self.complexity_labels.count(label) for label in self.complexity_labels],
            dtype=torch.float
        )


    def compute_energy_costs(self):
        sample_input = torch.randn(1, 256, 32, 32).to(self.device)
        energy_costs = [thop.profile(model, inputs=(sample_input,), verbose=False)[0] for model in [self.model.classifier1, self.model.classifier2, self.model.classifier3]]
        max_cost = max(energy_costs)
        return [cost / max_cost for cost in energy_costs]

    def compute_entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        max_entropy = np.log(probs.shape[-1])
        return entropy / max_entropy  
    
    def compute_sample_metrics(self):
        entropies, confidences, complexity_scores, complexity_labels = [], [], [], []
        self.model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(len(self.cifar10))):
                image, label = self.cifar10[i]
                image = image.unsqueeze(0).to(self.device)
                label = torch.tensor([label]).to(self.device)
                
                feat, [f1, o1], [f2, o2], [f3, o3] = self.model(image)
                outputs = [o1, o2, o3]
                
                complexities = []
                for idx, logits in enumerate(outputs):
                    probs = torch.softmax(logits, dim=-1)
                    confidence = probs[0, label].item()
                    entropy = self.compute_entropy(logits).item()
                    complexity_score = (1 - confidence) + self.alpha * entropy + self.beta * self.energy_costs[idx]
                    complexities.append(complexity_score)
                
                selected_module = np.argmin(complexities)
                complexity_labels.append(selected_module)
                complexity_scores.append(torch.tensor(complexities, dtype=torch.float32))
                
        return entropies, confidences, complexity_scores, complexity_labels
    
    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, self.complexity_labels[index], self.complexity_scores[index]
    
    def __len__(self):
        return len(self.cifar10)




    
def evaluate_model(args, task_model, loader, criterion):

    task_model.eval()
    task_model.to(args.device)

    running_loss_1 = 0.0
    running_loss_2 = 0.0
    running_loss_3 = 0.0

    running_corrects_1 = 0
    running_corrects_2 = 0
    running_corrects_3 = 0


    total_labels = 0
    since = time.time()
    for data in loader:
        inputs, labels, _, _ = data
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        
        with torch.no_grad():
            feat, [f1, x1], [f2, x2], [f3, x3] = task_model(inputs)

        loss_1 = criterion(x1, labels).item()
        loss_2 = criterion(x2, labels).item()
        loss_3 = criterion(x3, labels).item()

        _, pred1 = torch.max(x1, 1)
        _, pred2 = torch.max(x2, 1)
        _, pred3 = torch.max(x3, 1)

        running_loss_1 += loss_1 * inputs.size(0)
        running_loss_2 += loss_2 * inputs.size(0)
        running_loss_3 += loss_3 * inputs.size(0)

        total_labels += labels.size(0)

        running_corrects_1 += torch.sum(pred1 == labels.data).item()
        running_corrects_2 += torch.sum(pred2 == labels.data).item()
        running_corrects_3 += torch.sum(pred3 == labels.data).item()


    time_elapsed = time.time() - since

    eval_loss_1 = running_loss_1 / total_labels
    eval_accuracy_1 = running_corrects_1 / total_labels

    eval_loss_2 = running_loss_2 / total_labels
    eval_accuracy_2 = running_corrects_2 / total_labels

    eval_loss_3 = running_loss_3 / total_labels
    eval_accuracy_3 = running_corrects_3 / total_labels
    
    return [eval_loss_1, eval_loss_2, eval_loss_3], [eval_accuracy_1, eval_accuracy_2, eval_accuracy_3], time_elapsed



def compute_flops(model, input_tensor):
    macs, _ = thop.profile(model, inputs=(input_tensor,), verbose=False)
    return 2 * macs

def precompute_flops(task_model, sampler, device):
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    feature_map = task_model.feature_extractor(dummy_input)

    flops_feat = compute_flops(task_model.feature_extractor, dummy_input)
    flops_sampler = compute_flops(sampler, feature_map)
    
    flops_classifier1 = compute_flops(task_model.classifier1, feature_map)
    flops_classifier2 = compute_flops(task_model.classifier2, feature_map)
    flops_classifier3 = compute_flops(task_model.classifier3, feature_map)

    return flops_feat, flops_sampler, flops_classifier1, flops_classifier2, flops_classifier3


def evaluate_task(args, sampler, task_model, loader):
    flops_fe, flops_sampler, flops_classifier1, flops_classifier2, flops_classifier3 = precompute_flops(task_model, sampler, args.device)
    
    task_model.eval()
    sampler.eval()

    correct, total = 0, 0
    total_inference_time, total_flops = 0.0, 0
    start_time = time.time()

    with torch.no_grad():
        for images, labels, _, _ in tqdm(loader):
            images, labels = images.to(args.device), labels.to(args.device)

            features, (_, logits1), (_, logits2), (_, logits3) = task_model(images)

            _, sampler_outputs = sampler(features)
            sampler_decision = torch.argmax(sampler_outputs, dim=1)

            logits_list = torch.stack([logits1, logits2, logits3], dim=1)
            logits = logits_list[torch.arange(images.size(0)), sampler_decision]

            total_flops += (flops_fe + flops_sampler) * labels.size(0)  # Feature extractor FLOPs for all samples
            total_flops += torch.sum(
                torch.tensor([flops_classifier1, flops_classifier2, flops_classifier3], device=args.device)[sampler_decision]
            ).item()

            correct += torch.sum(torch.argmax(logits, 1) == labels).item()
            total += labels.size(0)


    total_inference_time = time.time() - start_time
    accuracy = correct / total
    return accuracy, total_inference_time, total_flops, flops_sampler

def evaluate_sampler(args, sampler, task_model, loader, criterion):

    task_model.eval()
    sampler.eval()
    task_model.to(args.device)

    running_loss = 0.0

    running_corrects = 0


    total_labels = 0
    since = time.time()
    for data in tqdm(loader):
        inputs, _, labels, _ = data
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        
        with torch.no_grad():
            z, _, _, _ = task_model(inputs)
            _, logits = sampler(z)

        loss = criterion(logits, labels).item()

        _, pred = torch.max(logits, 1)

        running_loss += loss * inputs.size(0)

        total_labels += labels.size(0)

        running_corrects += torch.sum(pred == labels.data).item()


    time_elapsed = time.time() - since

    eval_loss = running_loss / total_labels

    eval_accuracy = running_corrects / total_labels
    
    return eval_loss, eval_accuracy, time_elapsed


def train_sampler(args, task_model, sampler, train_dataloader, val_dataloader, optimizer_sampler, scheduler_sampler, logger, wandb):
    task_model.to(args.device)
    sampler.to(args.device)

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    since = time.time()
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_wt = copy.deepcopy(sampler.state_dict())

    for epoch in range(args.epochs):
        running_assignment_loss = 0.0
        running_feat_loss = 0.0
        running_complexity_loss = 0.0
        running_total_loss = 0.0
        running_corrects = 0.0

        sampler.train()
        task_model.eval()

        for data in tqdm(train_dataloader):
            imgs, task_labels, sampler_label, complexity_scores = data
            imgs, task_labels, sampler_label, complexity_scores = (
                imgs.to(args.device),
                task_labels.to(args.device),
                sampler_label.to(args.device),
                complexity_scores.to(args.device)
            )

            optimizer_sampler.zero_grad()

            with torch.no_grad():
                z, [f1, x1], [f2, x2], [f3, x3] = task_model(imgs)
            
            sampler_feat, sampler_pred = sampler(z)

            assignment_loss = ce_loss(sampler_pred, sampler_label)
            feat_loss = kl_loss(F.log_softmax(sampler_feat / args.T, dim=1), F.softmax(f1 / args.T, dim=1)) * (args.T * args.T)
            
            # Compute complexity loss
            probs = F.softmax(sampler_pred, dim=1)
            log_probs = torch.log(probs + 1e-10)
            complexity_loss = torch.sum(probs * (complexity_scores + args.gamma * log_probs), dim=1).mean()
            
            total_loss = assignment_loss + args.lambda_1 * feat_loss + args.lambda_2 * complexity_loss            
            
            # # Print gradients
            # for name, param in sampler.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradient for {name}: {param.grad.norm().item()}")
            # optimizer_sampler.zero_grad()
            
            total_loss.backward() 
            optimizer_sampler.step()

            _, spred = torch.max(sampler_pred, 1)
            running_assignment_loss += assignment_loss.item() * imgs.size(0)
            running_feat_loss += feat_loss.item() * imgs.size(0)
            running_complexity_loss += complexity_loss.item() * imgs.size(0)
            running_total_loss += total_loss.item() * imgs.size(0)
            running_corrects += torch.sum(spred == sampler_label.data).item()

        train_assignment_loss = running_assignment_loss / len(train_dataloader.dataset)
        train_feat_loss = running_feat_loss / len(train_dataloader.dataset)
        train_complexity_loss = running_complexity_loss / len(train_dataloader.dataset)
        train_total_loss = running_total_loss / len(train_dataloader.dataset)
        train_accuracy = running_corrects / len(train_dataloader.dataset)

        scheduler_sampler.step()

        val_loss, val_accuracy, _ = evaluate_sampler(args, sampler, task_model, val_dataloader, ce_loss)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_train_acc = train_accuracy
            best_wt = copy.deepcopy(sampler.state_dict())

        log_msg = {
            "Epoch": epoch + 1,
            "Train Total Loss": train_total_loss,
            "Train Routing Loss": train_assignment_loss,
            'Train Feature Loss': train_feat_loss,
            "Train Complexity Loss": train_complexity_loss,
            "Train Accuracy": train_accuracy * 100,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy * 100
        }


        logger.info(log_msg)
        if wandb:
            wandb.log(log_msg)

    time_elapsed = time.time() - since
    final_log = {
        "Training Time": time_elapsed,
        "Best Validation Accuracy": best_val_acc * 100,
        "Best Train Accuracy": best_train_acc * 100
    }
    sampler.load_state_dict(best_wt)
    return final_log, sampler
