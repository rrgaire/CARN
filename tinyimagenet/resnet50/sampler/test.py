import argparse
import torch
import torchvision
import torch.nn as nn
import wandb
import os
import logging
import torch_pruning as tp

from utils import *
from models import *

def get_args():
    parser = argparse.ArgumentParser(description='Train multiple models on CIFAR-10')
    parser.add_argument('--project', type=str, default='Complexity-aware Router', help='Name of the project')
    parser.add_argument('--entity', type=str, default='IDEA', help='Name of the entity')
    parser.add_argument('--name', type=str, default='test', help='Name of the experient')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--data_path', type=str, default="data", help='Path to save data') 
    parser.add_argument('--num_classes', type=int, default=10, help='Number of Classes') 
    parser.add_argument('--lr_task', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split percentage')
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
    parser.add_argument('--device', type=int, default=1 if torch.cuda.is_available() else 'cpu', help='Training device')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory to save checkpoints')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')  # Add this line

    return parser.parse_args()


class TinyImageNet(torch.utils.data.Dataset):

    def __init__(self, args, train=True, transform=None):
        self.data = torchvision.datasets.ImageFolder('../../dataset/val', transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    

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
    for data in tqdm(loader):
        inputs, labels = data
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


def get_model_info(model, input):

    feature_map = model.feature_extractor(input)
    macs_fe, params_fe = tp.utils.count_ops_and_params(model.feature_extractor, input)
    flops_fe = 2 * macs_fe

    macs_c1, params_c1 = tp.utils.count_ops_and_params(model.classifier1, feature_map)
    macs_c2, params_c2 = tp.utils.count_ops_and_params(model.classifier2, feature_map)
    macs_c3, params_c3 = tp.utils.count_ops_and_params(model.classifier3, feature_map)

    flops_c1 = 2 * macs_c1
    flops_c2 = 2 * macs_c2
    flops_c3 = 2 * macs_c3

    return [flops_fe, flops_c1, flops_c2, flops_c3], [params_fe, params_c1, params_c2, params_c3] 

def evaluate_task(args, sampler, task_model, loader, input):

    feature_map = task_model.feature_extractor(input)
    macs_sampler, params_sampler = tp.utils.count_ops_and_params(sampler, feature_map)
    flops_sampler = 2 * macs_sampler

    flops, params = get_model_info(task_model, input)
    
    task_model.eval()
    sampler.eval()

    correct, total = 0, 0
    total_inference_time, total_flops, total_params = 0.0, 0, 0
    start_time = time.time()

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(args.device), labels.to(args.device)

            features, (_, logits1), (_, logits2), (_, logits3) = task_model(images)

            _, sampler_outputs = sampler(features)
            sampler_decision = torch.argmax(sampler_outputs, dim=1)

            logits_list = torch.stack([logits1, logits2, logits3], dim=1)
            logits = logits_list[torch.arange(images.size(0)), sampler_decision]

            total_flops += (flops[0]) * labels.size(0)  # Feature extractor FLOPs for all samples
            total_flops += torch.sum(
                torch.tensor([flops[1], flops[3], flops[3]], device=args.device)[sampler_decision]
            ).item()

            total_params += (params[0]) * labels.size(0)
            total_params += torch.sum(
                torch.tensor([params[1], params[3], params[3]], device=args.device)[sampler_decision]
            ).item()
            correct += torch.sum(torch.argmax(logits, 1) == labels).item()
            total += labels.size(0)


    total_inference_time = time.time() - start_time
    accuracy = correct / total
    return accuracy, total_inference_time, total_flops, total_params, flops_sampler, params_sampler


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


    # fe = torch.load('/home/rrgaire/projects/iccv/expts/vgg19/cifar10/indv/checkpoints/vgg19_fe.pth')
    # module_1 = torch.load('/home/rrgaire/projects/iccv/expts/vgg19/cifar10/indv/checkpoints/vgg19_classifier.pth')
    # module_2 = torch.load('/home/rrgaire/projects/iccv/paper/cifar10/vgg19/pruning_v2/checkpoints/C10_VGG19_ratio_0.5_4_classifier.pth')
    # module_3 = torch.load('/home/rrgaire/projects/iccv/paper/cifar10/vgg19/pruning_v2/checkpoints/C10_VGG19_ratio_0.9_4_classifier.pth')

    # task_model = TaskModel(fe, module_1, module_2, module_3, False)
    task_model = torch.load('/home/rrgaire/projects/iccv/paper/tinyimagenet/resnet50/sampler/checkpoints/task_model.pth')
    task_model = task_model.to(args.device)

    sampler = Sampler(num_modules=3)
    sampler.load_state_dict(torch.load('/home/rrgaire/projects/iccv/paper/tinyimagenet/resnet50/sampler/checkpoints/TI_RNET_RFC__sampler.pth'))
    
    sampler = sampler.to(args.device)
    sampler.eval()
    task_model.eval()

    

    log_msg = {}
    

    example_inputs = torch.randn(1, 3, 32, 32).to(args.device)

    flops, params = get_model_info(task_model, example_inputs)


    normalize = torchvision.transforms.Normalize(
        mean = [0.49139968, 0.48215841, 0.44653091],
        std = [0.24703223, 0.24348513, 0.26158784]
    )
    
    test_transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize([32, 32]),
                    torchvision.transforms.ToTensor(),
                    normalize
    ])

    test_dataset = TinyImageNet(
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


    loss, acc, _ = evaluate_model(args, task_model, test_loader, nn.CrossEntropyLoss())

    log_msg["Task Module 1, Accuracy"] = acc[0] * 100
    log_msg["Task Module 1, FLOPs"] = ((flops[0] + flops[1]) * 10000) / 1e9
    log_msg["Task Module 1, Params"] = ((params[0] + params[1])* 10000) / 1e6
    log_msg["Task Module 2, Accuracy"] = acc[1] * 100
    log_msg["Task Module 2, FLOPs"] = ((flops[0] + flops[2]) * 10000) / 1e9
    log_msg["Task Module 2, Params"] = ((params[0] + params[2])* 10000) / 1e6
    log_msg["Task Module 3, Accuracy"] = acc[2] * 100
    log_msg["Task Module 3, FLOPs"] = ((flops[0] + flops[3]) * 10000) / 1e9
    log_msg["Task Module 3, Params"] = ((params[0] + params[3])* 10000) / 1e6
    

    

    task_accuracy, task_inference_time, task_flops, task_params, flops_sampler, params_sampler = evaluate_task(
            args=args,
            sampler=sampler,
            task_model=task_model,
            loader=test_loader,
            input=example_inputs
        )
    
    log_msg["Task Inference Time"] = task_inference_time
    log_msg["Task Test Accuracy"] = task_accuracy * 100
    log_msg["Task Task FLOPs"] = task_flops / 1e9
    log_msg["Task Task Params"] = task_params / 1e6
    log_msg["Sampler FLOPS"] = flops_sampler / 1e9
    log_msg["Sampler Params"] = params_sampler / 1e6


    
    
    # test_dataset = CIFAR10WithSampler(
    #         args=args,
    #         model=task_model,
    #         transform=test_transform,
    #         train=False
    #     )

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, 
    #     batch_size=args.batch_size,
    #     shuffle=False, 
    #     num_workers=8, 
    #     pin_memory=True
    # )

    # sampler_eval_loss, sampler_eval_accuracy, sampler_time_elapsed = evaluate_sampler(
    #         args=args,
    #         sampler=sampler,
    #         task_model=task_model,
    #         loader=test_loader,
    #         criterion=criterion
    #     )
    
    # log_msg["Sampler Inference Time"] = sampler_time_elapsed
    # log_msg["Sampler Test Accuracy"] = sampler_eval_accuracy * 100
    # log_msg["Sampler Test Loss"] = sampler_eval_loss

    



    logger.info(log_msg)
    if args.wandb:
        wandb.log(log_msg)
        wandb.finish()
    
if __name__ == '__main__':
    main()
