
import os
import argparse
import wandb
import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision
from torch.utils.data import DataLoader

from models import *
from utils import *

class MySlimmingPruner(tp.pruner.MetaPruner):
    def regularize(self, model, reg):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine==True:
                m.weight.grad.data.add_(reg*torch.sign(m.weight.data))

class MySlimmingImportance(tp.importance.Importance):
    def __call__(self, group, **kwargs):
        #note that we have multiple BNs in a group, 
        # we store layer-wise scores in a list and then reduce them to get the final results
        group_imp = [] # (num_bns, num_channels) 
        # 1. iterate the group to estimate importance
        for dep, idxs in group:
            layer = dep.target.module # get the target model
            prune_fn = dep.handler    # get the pruning function of target model, unused in this example
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and layer.affine:
                local_imp = torch.abs(layer.weight.data)
                group_imp.append(local_imp)
        if len(group_imp)==0: return None # return None if the group contains no BN layer
        # 2. reduce your group importance to a 1-D scroe vector. Here we use the average score across layers.
        group_imp = torch.stack(group_imp, dim=0).mean(dim=0) 
        return group_imp # (num_channels, )

def get_args():
    parser = argparse.ArgumentParser(description='Train multiple models on CIFAR-10')
    parser.add_argument('--project', type=str, default='Complexity-aware Router', help='Name of the project')
    parser.add_argument('--entity', type=str, default='IDEA', help='Name of the entity')
    parser.add_argument('--name', type=str, default='test', help='Name of the experient')
    parser.add_argument('--ratio', type=float, default=0.5, help='prune ratio')
    parser.add_argument('--epochs', type=int, default=10, help='Number of pre-training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--data_path', type=str, default="data", help='Path to save data') 
    parser.add_argument('--num_classes', type=int, default=10, help='Number of Classes') 
    parser.add_argument('--lr_task', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split percentage')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--step_size', type=int, default=10, help='Scheduler step size')
    parser.add_argument('--seed', type=int, default=2025, help='Seed value')
    parser.add_argument('--device', type=int, default=1 if torch.cuda.is_available() else 'cpu', help='Training device')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory to save checkpoints')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')  

    return parser.parse_args()
def main():
    args = get_args()

    set_random_seed(args.seed)
    args.name = args.name + f'_ratio_{args.ratio}'

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
            mean = [0.5071, 0.4865, 0.4409],
            std = [0.2673, 0.2564, 0.2762]
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

    criterion = nn.CrossEntropyLoss()
    example_inputs = torch.randn(1, 128, 16, 16)
    example_inputs = example_inputs.to(args.device)

    fe = torch.load('/home/rrgaire/projects/iccv/paper/cifar10/densenet121/indv/checkpoints/densenet121_fe.pth', map_location='cpu')
    vgg19 = torch.load('/home/rrgaire/projects/iccv/paper/cifar10/densenet121/indv/checkpoints/densenet121_classifier.pth', map_location='cpu')
    fe = fe.to(args.device)
    vgg19 = vgg19.to(args.device)
    for param in fe.parameters():
        param.requires_grad = False

    iterative_steps = 5

    # imp = MySlimmingImportance()
    imp = tp.importance.GroupNormImportance(p=2) 
    ignored_layers = []
    for m in vgg19.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == args.num_classes:
            ignored_layers.append(m) # DO NOT prune the final classifier!

    pruner = MySlimmingPruner(
        vgg19, 
        example_inputs, 
        global_pruning=False, # If False, a uniform pruning ratio will be assigned to different layers.
        importance=imp, # importance criterion for parameter selection
        iterative_steps=iterative_steps, # the number of iterations to achieve target pruning ratio
        pruning_ratio=args.ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(vgg19, example_inputs)

    for i in range(iterative_steps):
        pruner.step()

        macs, nparams = tp.utils.count_ops_and_params(vgg19, example_inputs)
        logger.info(
            "  Iter %d/%d, Params: %.2f M => %.2f M"
            % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
        )
        logger.info(
            "  Iter %d/%d, MACs: %.2f M => %.2f M"
        % (i+1, iterative_steps, base_macs / 1e6, macs / 1e6)
        )

        task_model = TaskModel(fe, vgg19, False)
        
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, task_model.parameters()), lr=args.lr_task, weight_decay=args.weight_decay, momentum=args.momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        
        log_msg, task_model = train_models(args, task_model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, logger, wandb if args.wandb else None, pruner)



        logger.info('Finetuning complete!')

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

        task_model.zero_grad() # Remove gradients
        torch.save(task_model.classifier, f'{args.checkpoint_dir}/{args.name}_{i}_classifier.pth') # without .state_dict

    logger.info("="*64)

    
if __name__ == '__main__':
    main()