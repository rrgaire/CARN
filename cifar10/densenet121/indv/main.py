import argparse
import torch
import torchvision
import torch.nn as nn
import wandb
import os
import random
import logging
import numpy as np

from models import *
from utils import *

def get_args():
    parser = argparse.ArgumentParser(description='Train multiple models on CIFAR-10')
    parser.add_argument('--project', type=str, default='Complexity-aware Router', help='Name of the project')
    parser.add_argument('--entity', type=str, default='IDEA', help='Name of the entity')
    parser.add_argument('--name', type=str, default='test', help='Name of the experient')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--data_path', type=str, default="data", help='Path to save data') 
    parser.add_argument('--num_classes', type=int, default=10, help='Number of Classes') 
    parser.add_argument('--lr_task', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split percentage')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--step_size', type=int, default=80, help='Scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='Scheduler gamma')
    parser.add_argument('--seed', type=int, default=2025, help='Seed value')
    parser.add_argument('--device', type=int, default=1 if torch.cuda.is_available() else 'cpu', help='Training device')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory to save checkpoints')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')  # Add this line

    return parser.parse_args()

def main():
    args = get_args()

    set_random_seed(args.seed)

    if args.wandb:
        wandb.login()
        wandb.init(
            project=args.project, 
            name=args.name, 
            config=vars(args), 
            notes=''
        )

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    setup_logger('base', args.name, args.log_dir, level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(vars(args))
    
    normalize = torchvision.transforms.Normalize(
        mean = [0.49139968, 0.48215841, 0.44653091],
        std = [0.24703223, 0.24348513, 0.26158784]
    )
    

    train_transform = torchvision.transforms.Compose([
    
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize,
       ])
    
    test_transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize([32, 32]),
                    torchvision.transforms.ToTensor(),
                    normalize
    ])

    test_dataset = CIFAR10(
            args=args,
            transform=test_transform,
            train=False
        )
    
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )
    
    train_dataset = CIFAR10(
        args=args,
        transform=train_transform,
        train=True
    )

    val_dataset = CIFAR10(
        args=args,
        transform=test_transform, 
        train=True
    )

    all_indices = list(np.arange(len(train_dataset)))
    val_size = int(args.val_split * len(train_dataset))
    val_indices = random.sample(all_indices, val_size)
    train_indices = np.setdiff1d(all_indices, val_indices)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)


    train_dataloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        drop_last=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False, 
        drop_last=True
    )
    
    fe = CommonFE(Bottleneck, 6)
    densenet121 = DenseNet121(num_classes=args.num_classes)
    # fe.load_state_dict(torch.load('/home/rrgaire/projects/iccv/expts/resnet50/indv/checkpoints/resnet50_feature_extractor.pth'))
    # densenet121.load_state_dict(torch.load('/home/rrgaire/projects/iccv/expts/resnet50/indv/checkpoints/resnet50_classifier.pth'))
    
    task_model = TaskModel(fe, densenet121, True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(task_model.parameters(), lr=args.lr_task, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_msg = {}
    log_msg, task_model = train_models(args, task_model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, logger, wandb if args.wandb else None)

    

    logger.info('Training complete!')
    torch.save(task_model.feature_extractor, f'{args.checkpoint_dir}/{args.name}_fe.pth')
    torch.save(task_model.classifier, f'{args.checkpoint_dir}/{args.name}_classifier.pth')

    task_model.eval()

    eval_loss, eval_accuracy, time_elapsed = evaluate_model(
            args=args,
            task_model=task_model,
            loader=test_loader,
            criterion=criterion
        )
    
    
    log_msg["Inference Time"] = time_elapsed
    log_msg["Test Accuracy"] = eval_accuracy * 100
    log_msg["Test Loss"] = eval_loss

    logger.info(log_msg)
    if args.wandb:
        wandb.log(log_msg)
        wandb.finish()
    
if __name__ == '__main__':
    main()
