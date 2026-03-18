import argparse
import torch
import torchvision
import torch.nn as nn
import wandb
import os
import random
import logging
import numpy as np
from collections import Counter

from models import *
from utils import *

def get_args():
    parser = argparse.ArgumentParser(description='Train multiple models on CIFAR-10')
    parser.add_argument('--project', type=str, default='Complexity-aware Router', help='Name of the project')
    parser.add_argument('--entity', type=str, default='IDEA', help='Name of the entity')
    parser.add_argument('--name', type=str, default='test', help='Name of the experient')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--data_path', type=str, default="data", help='Path to save data') 
    parser.add_argument('--num_classes', type=int, default=200, help='Number of Classes') 
    parser.add_argument('--lr_task', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.05, help='Validation split percentage')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--step_size', type=int, default=40, help='Scheduler step size')
    parser.add_argument('--alpha', type=float, default=0.6, help='Scheduler gamma')
    parser.add_argument('--beta', type=float, default=0.2, help='Scheduler gamma')
    parser.add_argument('--gamma', type=float, default=0.1, help='Scheduler gamma')
    parser.add_argument('--lambda_1', type=float, default=10.0, help='loss Weight lambda 1')
    parser.add_argument('--T', type=float, default=1.0, help='loss Weight lambda 1')
    parser.add_argument('--lambda_2', type=float, default=0.01, help='loss Weight lambda 1')
    parser.add_argument('--seed', type=int, default=2025, help='Seed value')
    parser.add_argument('--device', type=int, default=0 if torch.cuda.is_available() else 'cpu', help='Training device')
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


    fe = torch.load('/home/rrgaire/projects/iccv/paper/tinyimagenet/densenet121/indv/checkpoints/TI_DNET_INDV_fe.pth', map_location='cpu')
    densenet121_1 = torch.load('/home/rrgaire/projects/iccv/paper/tinyimagenet/densenet121/indv/checkpoints/TI_DNET_INDV_classifier.pth', map_location='cpu')
    densenet121_2 = torch.load('/home/rrgaire/projects/iccv/paper/tinyimagenet/densenet121/pruning_v2/checkpoints/TI_DNET_ratio_0.5_4_classifier.pth', map_location='cpu')
    densenet121_3 = torch.load('/home/rrgaire/projects/iccv/paper/tinyimagenet/densenet121/pruning_v2/checkpoints/TI_DNET_ratio_0.9_3_classifier.pth', map_location='cpu')

    task_model = TaskModel(fe, densenet121_1, densenet121_2, densenet121_3, False)
    torch.save(task_model, f'{args.checkpoint_dir}/task_model.pth')         
    
    task_model.to(args.device)

    
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    

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

    test_dataset = TinyImageNetWithSampler(
            args=args,
            model=task_model,
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
    loss, acc, _ = evaluate_model(args, task_model, test_loader, nn.CrossEntropyLoss())
    logger.info(acc)

    train_dataset = TinyImageNetWithSampler(
        args=args,
        model=task_model,
        transform=train_transform,
        train=True
    )

    val_dataset = TinyImageNetWithSampler(
        args=args,
        model=task_model,
        transform=test_transform, 
        train=True
    )


    all_indices = list(np.arange(len(train_dataset)))
    val_size = int(args.val_split * len(train_dataset))
    val_indices = random.sample(all_indices, val_size)
    train_indices = np.setdiff1d(all_indices, val_indices)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    train_weights = torch.tensor(train_dataset.weights.clone().detach())[train_indices]

    t_sampler = torch.utils.data.WeightedRandomSampler(
    weights=train_weights,
    num_samples=len(train_weights),
    replacement=True 
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_subset,
        sampler=t_sampler,
        batch_size=args.batch_size,
        drop_last=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False, 
        drop_last=True
    )


    
    sampler = Sampler(num_modules=3)
    sampler.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(sampler.parameters(), lr=args.lr_task, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    log_msg, sampler = train_sampler(args, task_model, sampler, train_dataloader, val_dataloader, optimizer, scheduler, logger, wandb if args.wandb else None)



    logger.info('Training complete!')

    sampler.eval()

    torch.save(sampler.state_dict(), os.path.join(args.checkpoint_dir, f'{args.name}_sampler.pth'))

    sampler_eval_loss, sampler_eval_accuracy, sampler_time_elapsed = evaluate_sampler(
            args=args,
            sampler=sampler,
            task_model=task_model,
            loader=test_loader,
            criterion=criterion
        )
    
    task_accuracy, task_inference_time, task_flops, flops_sampler = evaluate_task(
            args=args,
            sampler=sampler,
            task_model=task_model,
            loader=test_loader,
        )

    log_msg["Sampler Inference Time"] = sampler_time_elapsed
    log_msg["Sampler Test Accuracy"] = sampler_eval_accuracy * 100
    log_msg["Sampler Test Loss"] = sampler_eval_loss

    log_msg["Task Inference Time"] = task_inference_time
    log_msg["Task Test Accuracy"] = task_accuracy * 100
    log_msg["Task Task FLOPs"] = task_flops
    log_msg["Sampler FLOPS"] = flops_sampler



    logger.info(log_msg)
    if args.wandb:
        wandb.log(log_msg)
        wandb.finish()
    
if __name__ == '__main__':
    main()
