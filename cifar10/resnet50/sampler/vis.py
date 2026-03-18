import torch
import argparse
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns

def imshow(img, title, save_path):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis("off")
    plt.savefig(save_path)
    plt.show()

def visualize_dataset(dataset, complexities, save_path="dataset_visualization.png", num_samples=16):
    """Visualize and save CIFAR10 samples with complexity labels and breakdown."""
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    fig.suptitle("CIFAR10 Samples with Complexity Breakdown", fontsize=16)
    
    for i in range(num_samples):
        image, _ = dataset.cifar10[i]
        complexity = complexities[i]
        
        # Image
        img = image / 2 + 0.5  # Unnormalize
        npimg = np.transpose(img.numpy(), (1, 2, 0))
        
        axes[i, 0].imshow(npimg)
        axes[i, 0].set_title(f"Sample #{i} - Complexity: {sum(complexity):.2f}")
        axes[i, 0].axis("off")
        
        # Bar Graph (Complexity Breakdown)
        labels = ['Energy Inefficiency', 'Entropy', 'Error (1 - Accuracy)']
        colors = ['green', 'orange', 'blue']
        
        axes[i, 1].bar(labels, complexity, color=colors)
        axes[i, 1].set_ylabel("Complexity Score")
        axes[i, 1].set_title("Complexity Breakdown")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def visualize_tsne(model, dataloader, device, save_path="tsne_plot.png"): 
    """Generate t-SNE plots for feature visualization."""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.numpy()
            
            feats = model.common_feature_extractor(images).cpu().numpy()
            features.append(feats)
            labels.append(targets)
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_features = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=labels, palette=sns.color_palette("hsv", 10), alpha=0.7)
    plt.title("t-SNE Visualization of Features")
    plt.legend(title="Classes")
    plt.savefig(save_path)
    plt.show()

# Example usage (assuming dataset, complexities, and dataloader are defined)
from utils import *
from models import *

def get_args():
    parser = argparse.ArgumentParser(description='Train multiple models on CIFAR-10')
    parser.add_argument('--project', type=str, default='Complexity-aware Router', help='Name of the project')
    parser.add_argument('--entity', type=str, default='IDEA', help='Name of the entity')
    parser.add_argument('--name', type=str, default='test', help='Name of the experient')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--data_path', type=str, default="data", help='Path to save data') 
    parser.add_argument('--num_classes', type=int, default=10, help='Number of Classes') 
    parser.add_argument('--lr_task', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split percentage')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--step_size', type=int, default=80, help='Scheduler step size')
    parser.add_argument('--alpha', type=float, default=0.8, help='Scheduler gamma')
    parser.add_argument('--beta', type=float, default=0.3, help='Scheduler gamma')
    parser.add_argument('--gamma', type=float, default=0.1, help='Scheduler gamma')
    parser.add_argument('--loss_wt', type=float, default=0.1, help='loss Weight')
    parser.add_argument('--seed', type=int, default=2025, help='Seed value')
    parser.add_argument('--device', type=int, default=0 if torch.cuda.is_available() else 'cpu', help='Training device')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory to save checkpoints')
    return parser.parse_args()

args = get_args()

fe = CommonFE()
vgg1 = VGG1(num_classes=10)
vgg2 = VGG2(num_classes=10)

task_model = TaskModel(fe, vgg1, vgg2, True)
task_model.load_state_dict(torch.load('/home/rrgaire/projects/iccv/expts/isvlsi/combined/checkpoints/com_isvlsi_200_model.pth'))
task_model = task_model.to(0)


normalize = torchvision.transforms.Normalize(
    mean = [0.49139968, 0.48215841, 0.44653091],
    std = [0.24703223, 0.24348513, 0.26158784]
)


train_transform = torchvision.transforms.Compose([

    torchvision.transforms.Resize([32, 32]),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize,
    ])

test_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize([32, 32]),
                torchvision.transforms.ToTensor(),
                normalize
])

test_dataset = CIFAR10WithSampler(
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

train_dataset = CIFAR10WithSampler(
    args=args,
    model=task_model,
    transform=train_transform,
    train=True
)

val_dataset = CIFAR10WithSampler(
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


class CIFAR10WithSampler(torch.utils.data.Dataset):
    def __init__(self, args, model, transform, train=True):
        """
        CIFAR10 dataset extended to include complexity labels based on accuracy, entropy, and energy efficiency.
        """
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=args.data_path,
            download=True,
            train=train,
            transform=transform
        )
        
        self.device = args.device
        self.model = model
        self.alpha = args.alpha
        self.beta = args.beta  # Weighting parameter for energy inefficiency
        
        self.energy_costs = self.compute_energy_costs()
        self.complexity_labels = self._generate_sampler_labels()

    def compute_entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # Add epsilon to avoid log(0)
    
    def compute_energy_costs(self):
        """Compute FLOPs for each task module as a proxy for energy inefficiency."""
        sample_input = torch.randn(1, 64, 16, 16).to(self.device)
        
        energy_costs = []
        for classifier in [self.model.classifier1, self.model.classifier2]:
            flops = FlopCountAnalysis(classifier, sample_input)
            energy_costs.append(flops.total())
        
        max_cost = max(energy_costs)  # Normalize by max FLOPs
        return [cost / max_cost for cost in energy_costs]
    
    def _generate_sampler_labels(self):
        complexity_labels = []
        self.model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(len(self.cifar10))):
                image, label = self.cifar10[i]
                image = image.unsqueeze(0).to(self.device)  # Add batch dimension
                label = torch.tensor([label]).to(self.device)
                
                _, l1, l2 = self.model(image)
                outputs = [l1, l2]
                complexities = []
                
                for idx, logits in enumerate(outputs):
                    predictions = torch.argmax(logits, dim=-1)
                    accuracies = (predictions == label).float()
                    entropies = self.compute_entropy(logits)
                    normalized_entropy = entropies / np.log(10)
                    
                    energy_inefficiency = self.energy_costs[idx]
                    
                    complexity_score = ((1 - accuracies) +
                                        self.alpha * normalized_entropy +
                                        self.beta * energy_inefficiency)
                    complexities.append(complexity_score)
                
                complexities = torch.stack(complexities).squeeze(1)  # Shape: [num_modules]
                selected_module = torch.argmin(complexities)  # Select module with lowest complexity
                complexity_labels.append(selected_module.item())
        
        return complexity_labels



visualize_dataset(train_dataset, train_dataset.complexity_labels, "cifar10_complexity.png")
visualize_tsne(task_model, test_loader, args.device, "tsne_plot.png")
