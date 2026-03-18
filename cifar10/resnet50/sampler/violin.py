import torch
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from main import *
from utils import *
from models import *

sns.set_theme(style="darkgrid")

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
        self.features, self.entropies, self.confidences, self.complexity_scores, self.complexity_labels = self.compute_sample_metrics(module)
    
    def compute_energy_costs(self):
        sample_input = torch.randn(1, 64, 16, 16).to(self.device)
        energy_costs = [FlopCountAnalysis(classifier, sample_input).total() for classifier in [self.model.classifier1, self.model.classifier2]]
        max_cost = max(energy_costs)
        return [cost / max_cost for cost in energy_costs]  # Normalize in [0,1]

    def compute_entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        max_entropy = np.log(probs.shape[-1])  # Normalizing by log(K) where K is num_classes
        return entropy / max_entropy  # Normalized entropy in [0,1]
    
    def compute_sample_metrics(self, module):
        features, entropies, confidences, complexity_scores, complexity_labels = [], [], [], [], []
        self.model.eval()
        
        with torch.no_grad():
            for i in range(len(self.cifar10)):
                image, label = self.cifar10[i]
                image = image.unsqueeze(0).to(self.device)
                label = torch.tensor([label]).to(self.device)
                
                feat, l1, l2 = self.model(image)
                
                outputs = [l1, l2]

                # for idx, logits in enumerate(outputs):
                logits = outputs[module]
                probs = torch.softmax(logits, dim=-1)
                confidence = probs[0, label].item()
                entropy = self.compute_entropy(logits).item()
                
                complexity_score = (1 - confidence) + self.alpha * entropy + self.beta * self.energy_costs[module]
                
                # selected_module = np.argmin(complexities)
                features.append(feat.detach().cpu().numpy())
                complexity_labels.append(module)
                complexity_scores.append(complexity_score)
                entropies.append(entropy)
                confidences.append(1 - confidence)

        return features, entropies, confidences, complexity_scores, complexity_labels
    def __len__(self):
        return len(self.cifar10)
    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, self.complexity_labels[index]

class CIFAR10WithSampler_(torch.utils.data.Dataset):
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
        self.features, self.entropies, self.confidences, self.complexity_scores, self.complexity_labels = self.compute_sample_metrics()
        
        self.weights = torch.tensor(
            [len(self.complexity_labels) / self.complexity_labels.count(label) for label in self.complexity_labels],
            dtype=torch.float
        )

    def compute_energy_costs(self):
        sample_input = torch.randn(1, 64, 16, 16).to(self.device)
        energy_costs = [FlopCountAnalysis(classifier, sample_input).total() for classifier in [self.model.classifier1, self.model.classifier2]]
        max_cost = max(energy_costs)
        return [cost / max_cost for cost in energy_costs]  # Normalize in [0,1]

    def compute_entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        max_entropy = np.log(probs.shape[-1])  # Normalizing by log(K) where K is num_classes
        return entropy / max_entropy  # Normalized entropy in [0,1]
    
    def compute_sample_metrics(self):
        features, entropies, confidences, complexity_scores, complexity_labels = [], [], [], [], []
        self.model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(len(self.cifar10))):
                image, label = self.cifar10[i]
                image = image.unsqueeze(0).to(self.device)
                label = torch.tensor([label]).to(self.device)
                
                feat, l1, l2 = self.model(image)
                
                outputs = [l1, l2]
                complexities = []
                sample_entropies = []
                sample_confidences = []
                
                for idx, logits in enumerate(outputs):
                    probs = torch.softmax(logits, dim=-1)
                    confidence = probs[0, label].item()
                    entropy = self.compute_entropy(logits).item()
                    sample_entropies.append(entropy)
                    sample_confidences.append(1 - confidence)
                    
                    complexity_score = (1 - confidence) + self.alpha * entropy + self.beta * self.energy_costs[idx]
                    complexities.append(complexity_score)
                
                selected_module = np.argmin(complexities)
                features.append(feat.detach().cpu().numpy())
                complexity_labels.append(selected_module)
                complexity_scores.append(complexities[selected_module])
                entropies.append(sample_entropies[selected_module])
                confidences.append(sample_confidences[selected_module])

        return features, entropies, confidences, complexity_scores, complexity_labels
    
    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, self.complexity_labels[index]
    
    def get_full_sample_data(self, index):
        data, target = self.cifar10[index]
        return data, target, (1 - self.confidences[index]), self.entropies[index], self.energy_costs[index]
    
    def __len__(self):
        return len(self.cifar10)

def plot_and_save_figure(dataset, dataset_0, dataset_1, save_path="cifar10_complexity_plot.png"):
    cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    sample_classes = [dataset.cifar10[i][1] for i in range(len(dataset))]
    complexity_labels = dataset.complexity_labels
    complexity_scores = dataset.complexity_scores

    df = pd.DataFrame({
        "class": [cifar10_classes[i] for i in sample_classes],
        "module": ["Large Module" if x == 0 else "Small Module" for x in complexity_labels],
        "complexity_score": complexity_scores
    })

    count_data = df.groupby(["class", "module"]).size().reset_index(name="count")
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 3]})
    
    sns.barplot(ax=axes[0], x="class", y="count", hue="module", data=count_data, palette="Pastel1")
    axes[0].set_ylabel("# Samples")
    axes[0].set_xlabel("")
    axes[0].legend(title="Modules", loc='upper right', bbox_to_anchor=(1.1, 1))
    
    sample_classes = [dataset_0.cifar10[i][1] for i in range(len(dataset_0))] + [dataset_1.cifar10[i][1] for i in range(len(dataset_1))]
    complexity_labels = dataset_0.complexity_labels + dataset_1.complexity_labels
    complexity_scores = dataset_0.complexity_scores + dataset_1.complexity_scores
    
    df = pd.DataFrame({
        "class": [cifar10_classes[i] for i in sample_classes],
        "module": ["Large Module" if x == 0 else "Small Module" for x in complexity_labels],
        "complexity_score": complexity_scores
    })
    
    sns.violinplot(ax=axes[1], x="class", y="complexity_score", hue="module", data=df, dodge=True, width=1.2, linewidth=1, palette="Pastel1", split=False)
    axes[1].set_ylabel("Complexity Score")
    axes[1].set_xlabel("CIFAR-10 Classes")
    
    # Set x-axis labels and rotation for both plots
    axes[0].set_xticklabels(cifar10_classes, rotation=45)
    axes[1].set_xticklabels(cifar10_classes, rotation=45)
    
    # Adjust layout and ensure both plots share the same legend
    axes[0].legend(title="Modules", loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# def plot_and_save_figure(dataset, dataset_0, dataset_1, save_path="cifar10_complexity_plot.png"):
#     cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
#     sample_classes = [dataset.cifar10[i][1] for i in range(len(dataset))]
#     complexity_labels = dataset.complexity_labels
#     complexity_scores = dataset.complexity_scores

#     df = pd.DataFrame({
#         "class": [cifar10_classes[i] for i in sample_classes],
#         "module": ["Large Module" if x == 0 else "Small Module" for x in complexity_labels],
#         "complexity_score": complexity_scores
#     })

#     count_data = df.groupby(["class", "module"]).size().reset_index(name="count")
    
#     fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 3]})
    
#     sns.barplot(ax=axes[0], x="class", y="count", hue="module", data=count_data, palette="Pastel1",)
#     axes[0].set_ylabel("# Samples")
#     axes[0].set_xlabel("")
#     axes[0].legend(title="Modules")

    
    
#     sample_classes = [dataset_0.cifar10[i][1] for i in range(len(dataset_0))] + [dataset_1.cifar10[i][1] for i in range(len(dataset_1))]
#     complexity_labels = dataset_0.complexity_labels + dataset_1.complexity_labels
#     complexity_scores = dataset_0.complexity_scores + dataset_1.complexity_scores
    
#     df = pd.DataFrame({
#         "class": [cifar10_classes[i] for i in sample_classes],
#         "module": ["Large Module" if x == 0 else "Small Module" for x in complexity_labels],
#         "complexity_score": complexity_scores
#     })
    
    
#     sns.violinplot(ax=axes[1], x="class", y="complexity_score", hue="module", data=df, dodge=True, width= 1.2, linewidth=1, palette="Pastel1", split=False) #, inner="quartile"
#     axes[1].set_ylabel("Complexity Score")
#     axes[1].set_xlabel("CIFAR-10 Classes")
#     axes[1].legend(title="Modules")
    
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.show()  

# def plot_and_save_figure(dataset, save_path="cifar10_complexity_plot.png"):
#     cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
#     sample_classes = [dataset.cifar10[i][1] for i in range(len(dataset))]
#     complexity_labels = dataset.complexity_labels
#     complexity_scores = dataset.complexity_scores
    
#     df = pd.DataFrame({
#         "class": [cifar10_classes[i] for i in sample_classes],
#         "module": ["Module 1" if x == 0 else "Module 2" for x in complexity_labels],
#         "complexity_score": complexity_scores
#     })
    
#     count_data = df.groupby(["class", "module"]).size().reset_index(name="count")
    
#     fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 3]})
    
#     sns.barplot(ax=axes[0], x="class", y="count", hue="module", data=count_data)
#     axes[0].set_ylabel("Count")
#     axes[0].set_xlabel("")
#     axes[0].legend(title="Module")
    
#     sns.violinplot(ax=axes[1], x="class", y="complexity_score", hue="module", data=df, split=True, inner="stick")
#     axes[1].set_ylabel("Complexity Score")
#     axes[1].set_xlabel("CIFAR-10 Classes")
#     axes[1].legend(title="Module")
    
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.show()



args = get_args()

save_dir = 'figures/train'
os.makedirs(save_dir, exist_ok=True)

fe = CommonFE()
vgg1 = VGG1(num_classes=args.num_classes)
vgg2 = VGG2(num_classes=args.num_classes)

task_model = TaskModel(fe, vgg1, vgg2, True)
task_model.load_state_dict(torch.load('/home/rrgaire/projects/iccv/expts/isvlsi/combined/checkpoints/com_isvlsi_200_model.pth'))
task_model = task_model.to(args.device)


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

test_dataset_0 = CIFAR10WithSampler(
        args=args,
        model=task_model,
        transform=test_transform,
        train=True,
        module=0
    )
test_dataset_1 = CIFAR10WithSampler(
        args=args,
        model=task_model,
        transform=test_transform,
        train=True,
        module=1
    )
test_dataset = CIFAR10WithSampler_(
    args=args,
    model=task_model,
    transform=test_transform,
    train=True
)

plot_and_save_figure(test_dataset, test_dataset_0, test_dataset_1, save_path="cifar10_complexity_plot.png")