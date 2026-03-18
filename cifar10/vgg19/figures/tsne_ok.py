import os
import torch
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from main import *
from utils import *
from models import *

# sns.set_theme(style="darkgrid")
cifar10_classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
task_module = ['VGG19_p00', 'VGG19_p50', 'VGG19_p90']
class CIFAR10WithSampler(torch.utils.data.Dataset):
    def __init__(self, args, model, transform, train=True, module=0):
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
        self.entropies, self.confidences, self.complexity_scores, self.complexity_labels = self.compute_sample_metrics(module)
        m_complex = max(self.complexity_scores)
        self.complexity_scores = [x / m_complex for x in self.complexity_scores]
        print(max(self.complexity_scores), min(self.complexity_scores))
    
    def compute_energy_costs(self):
        sample_input = torch.randn(1, 128, 8, 8).to(self.device)
        energy_costs = [thop.profile(model, inputs=(sample_input,), verbose=False)[0] for model in [self.model.classifier1, self.model.classifier2, self.model.classifier3]]
        max_cost = max(energy_costs)
        return [cost / max_cost for cost in energy_costs]

    def compute_entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        max_entropy = np.log(probs.shape[-1])
        return entropy / max_entropy  
    
    def compute_sample_metrics(self, module):
        entropies, confidences, complexity_scores, complexity_labels = [], [], [], []
        self.model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(len(self.cifar10))):
                image, label = self.cifar10[i]
                image = image.unsqueeze(0).to(self.device)
                label = torch.tensor([label]).to(self.device)
                
                feat, [f1, o1], [f2, o2], [f3, o3] = self.model(image)
                outputs = [o1, o2, o3]

                # for idx, logits in enumerate(outputs):
                logits = outputs[module]
                probs = torch.softmax(logits, dim=-1)
                confidence = probs[0, label].item()
                entropy = self.compute_entropy(logits).item()
                
                complexity_score = (1 - confidence) + self.alpha * entropy + self.beta * self.energy_costs[module]
                
                complexity_labels.append(module)
                complexity_scores.append(complexity_score)
                
        return entropies, confidences, complexity_scores, complexity_labels
    
    def __len__(self):
        return len(self.cifar10)
    
    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, self.complexity_labels[index]


class CIFAR10WithSampler_(torch.utils.data.Dataset):
    def __init__(self, args, model, sampler, transform, train=True):
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=args.data_path,
            download=True,
            train=train,
            transform=transform
        )
        
        self.device = args.device
        self.model = model
        self.sampler = sampler

        self.alpha = args.alpha
        self.beta = args.beta
        
        self.energy_costs = self.compute_energy_costs()
        self.features, self.entropies, self.confidences, self.complexity_scores, self.complexity_labels, self.task_labels, self.sampler_labels, self.sampler_features = self.compute_sample_metrics()
        
        self.weights = torch.tensor(
            [len(self.complexity_labels) / self.complexity_labels.count(label) for label in self.complexity_labels],
            dtype=torch.float
        )


    def compute_energy_costs(self):
        sample_input = torch.randn(1, 128, 8, 8).to(self.device)
        energy_costs = [thop.profile(model, inputs=(sample_input,), verbose=False)[0] for model in [self.model.classifier1, self.model.classifier2, self.model.classifier3]]
        max_cost = max(energy_costs)
        return [cost / max_cost for cost in energy_costs]

    def compute_entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        max_entropy = np.log(probs.shape[-1])
        return entropy / max_entropy  
    
    def compute_sample_metrics(self):
        # entropies, confidences, complexity_scores, complexity_labels = [], [], [], []
        features, entropies, confidences, complexity_scores, complexity_labels, task_labels, sampler_labels = [], [], [], [], [], [], []
        sampler_features = []
        self.model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(len(self.cifar10))):
                image, label = self.cifar10[i]
                image = image.unsqueeze(0).to(self.device)
                label = torch.tensor([label]).to(self.device)

                task_labels.append(label.item())
                
                feat, [f1, o1], [f2, o2], [f3, o3] = self.model(image)
                s_feat, sampler_logits = self.sampler(feat)
                _, s_preds = torch.max(sampler_logits, 1)
                sampler_labels.append(s_preds.item())

                outputs = [o1, o2, o3]
                
                complexities = []
                for idx, logits in enumerate(outputs):
                    probs = torch.softmax(logits, dim=-1)
                    confidence = probs[0, label].item()
                    entropy = self.compute_entropy(logits).item()
                    complexity_score = (1 - confidence) + self.alpha * entropy + self.beta * self.energy_costs[idx]
                    complexities.append(complexity_score)
                
                features.append(f1.detach().cpu().numpy())
                selected_module = np.argmin(complexities)
                complexity_labels.append(selected_module)
                complexity_scores.append(complexities)
                sampler_features.append(s_feat.detach().cpu().numpy())
        return features, entropies, confidences, complexity_scores, complexity_labels, task_labels, sampler_labels, sampler_features
    
    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, self.complexity_labels[index], self.complexity_scores[index]
    
    def __len__(self):
        return len(self.cifar10)



args = get_args()

save_dir = 'figures/train___'
os.makedirs(save_dir, exist_ok=True)

task_model = torch.load('/home/rrgaire/projects/iccv/paper/cifar10/vgg19/sampler/checkpoints/task_model.pth')
task_model = task_model.to(args.device)

sampler = Sampler(num_modules=3)
sampler.load_state_dict(torch.load('/home/rrgaire/projects/iccv/paper/cifar10/vgg19/sampler/checkpoints/C10_VGG19_RFC__sampler.pth'))

sampler = sampler.to(args.device)
sampler.eval()
task_model.eval()

normalize = torchvision.transforms.Normalize(
    mean = [0.49139968, 0.48215841, 0.44653091],
    std = [0.24703223, 0.24348513, 0.26158784]
)


test_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize([32, 32]),
                torchvision.transforms.ToTensor(),
                normalize
])


test_dataset = CIFAR10WithSampler_(
    args=args,
    model=task_model,
    sampler=sampler,
    transform=test_transform,
    train=True
)

def subset_samples(task_labels, features, complexity_labels, sampler_labels, fraction=4):
    """
    Selects a subset of the dataset while ensuring that the smallest task module is adequately represented.

    Parameters:
    - task_labels: Array of task module labels.
    - features: Feature representations.
    - complexity_labels: Complexity class labels.
    - sampler_labels: Sampler-predicted labels.
    - fraction: The fraction of samples to retain from the smallest class.

    Returns:
    - A subset of features, task_labels, complexity_labels, and sampler_labels.
    """
    unique_tasks, counts = np.unique(complexity_labels, return_counts=True)
    min_count = min(counts)  # Find the smallest task module count
    subset_size = int(min_count * fraction)  # Take a fraction of the smallest task module size

    selected_indices = []
    for task in unique_tasks:
        task_indices = np.where(complexity_labels == task)[0]
        subset_indices = np.random.choice(task_indices, min(subset_size, len(task_indices)), replace=False)
        selected_indices.extend(subset_indices)

    return (
        features[selected_indices],
        task_labels[selected_indices],
        complexity_labels[selected_indices],
        sampler_labels[selected_indices]
    )


def plot_tsne_pca(features, task_labels, shape_labels, save_dir, legend):
    """
    Plots t-SNE with task labels (color distinction) and another label (shape distinction).

    Parameters:
    - features: Feature vectors.
    - task_labels: Labels for color distinction.
    - shape_labels: Labels for marker distinction (complexity or sampler predictions).
    - save_dir: File path to save the plot.
    - title: Plot title.
    - shape_legend: Legend title for shape markers.
    """
    # Flatten and normalize features
    features = np.array([f.flatten() for f in features])
    features = StandardScaler().fit_transform(features)

    # Reduce dimensionality with PCA before t-SNE
    pca = PCA(n_components=50).fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(pca)

    # Define marker shapes for different shape labels
    # unique_shapes = np.unique(shape_labels)
    markers = ['o', 's', 'X']  # Circle, square, cross, diamond, triangle, etc.
    # marker_map = {label: markers[i % len(markers)] for i, label in enumerate(unique_shapes)}

    plt.figure(figsize=(3.5, 6.5))
    ax = sns.scatterplot(
        x=tsne[:, 0], 
        y=tsne[:, 1], 
        hue=shape_labels,  # Task labels for color
        style=shape_labels,  # Shape labels for markers
        markers=markers, 
        s=10, 
        # palette='tab10', 
        alpha=1
    )

    # Remove axis labels, ticks, and frame
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_frame_on(False)

    # Remove spines (border lines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # plt.title(title)
    if legend:
        plt.legend(title='Task Modules', loc='upper right', bbox_to_anchor=(1.15, 1))
    else:
        ax.get_legend().remove()
    plt.tight_layout()
    plt.savefig(save_dir, dpi=300, bbox_inches ='tight')
    plt.close()

plot_tsne_pca(
    test_dataset.features, 
    test_dataset.task_labels, 
    [task_module[l] for l in test_dataset.complexity_labels], 
    f'{save_dir}/tsne_pca_task_complexity.png', 
    legend= False
)

plot_tsne_pca(
    test_dataset.features, 
    test_dataset.task_labels, 
    [task_module[l] for l in test_dataset.sampler_labels], 
    f'{save_dir}/tsne_pca_task_sampler.png', 
    legend=True
)