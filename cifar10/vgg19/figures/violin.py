import os
import torch
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from main import *
from utils import *
from models import *

# sns.set_theme(style="darkgrid")

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
                complexity_scores.append(complexities)
                
        return entropies, confidences, complexity_scores, complexity_labels
    
    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, self.complexity_labels[index], self.complexity_scores[index]
    
    def __len__(self):
        return len(self.cifar10)

def plot_and_save_figure(dataset, dataset_0, dataset_1, dataset_2, save_path_bar="a.png", save_path_violin="b.png"):
    custom_palette = {"VGG19_p00": "#E24A33", "VGG19_p50": "#348ABD", "VGG19_p90": "#988ED5"}

    # Display only the first 5 classes
    cifar10_classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

    # Prepare data for bar plot
    sample_classes = [dataset.cifar10[i][1] for i in range(len(dataset)) if dataset.cifar10[i][1]]
    complexity_labels = [dataset.complexity_labels[i] for i in range(len(dataset)) if dataset.cifar10[i][1]]
    complexity_scores = [dataset.complexity_scores[i] for i in range(len(dataset)) if dataset.cifar10[i][1]]

    df_bar = pd.DataFrame({
        "class": [cifar10_classes[i] for i in sample_classes],
        "module": ["VGG19_p00" if x == 0 else "VGG19_p50" if x == 1 else "VGG19_p90" for x in complexity_labels],
        "complexity_score": complexity_scores
    })

    count_data = df_bar.groupby(["class", "module"]).size().reset_index(name="count")

    # Plot bar graph
    plt.figure(figsize=(10, 5))
    sns.barplot(x="class", y="count", hue="module", data=count_data)
    plt.ylabel("Number of Samples")
    plt.xlabel("")
    # plt.xticks(rotation=45)
    plt.legend(title="Task Modules", loc='upper right', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(save_path_bar, dpi=400, bbox_inches='tight')
    plt.show()

    # Prepare data for violin plot
    sample_classes = [dataset_0.cifar10[i][1] for i in range(len(dataset_0)) if dataset_0.cifar10[i][1] < 5] + \
                    [dataset_1.cifar10[i][1] for i in range(len(dataset_1)) if dataset_1.cifar10[i][1] < 5] + \
                    [dataset_2.cifar10[i][1] for i in range(len(dataset_2)) if dataset_2.cifar10[i][1] < 5]

    complexity_labels = [dataset_0.complexity_labels[i] for i in range(len(dataset_0)) if dataset_0.cifar10[i][1] < 5] + \
                       [dataset_1.complexity_labels[i] for i in range(len(dataset_1)) if dataset_1.cifar10[i][1] < 5] + \
                       [dataset_2.complexity_labels[i] for i in range(len(dataset_2)) if dataset_2.cifar10[i][1] < 5]

    complexity_scores = [dataset_0.complexity_scores[i] for i in range(len(dataset_0)) if dataset_0.cifar10[i][1] < 5] + \
                       [dataset_1.complexity_scores[i] for i in range(len(dataset_1)) if dataset_1.cifar10[i][1] < 5] + \
                       [dataset_2.complexity_scores[i] for i in range(len(dataset_2)) if dataset_2.cifar10[i][1] < 5]

    # # Normalization of complexity scores for better visualization
    # min_score, max_score = min(complexity_scores), max(complexity_scores)
    # normalized_scores = [(score - min_score) / (max_score - min_score) for score in complexity_scores]

    df_violin = pd.DataFrame({
        "class": [cifar10_classes[i] for i in sample_classes],
        "module": ["VGG19_p00" if x == 0 else "VGG19_p50" if x == 1 else "VGG19_p90" for x in complexity_labels],
        "complexity_score": complexity_scores,
    })

    # Plot violin graph with adjusted width and separation
    plt.figure(figsize=(10, 5))
    sns.violinplot(x="class", y="complexity_score", hue="module", data=df_violin, inner_kws=dict(box_width=15, whis_width=2, color=".8"), )
    plt.ylabel("Complexity Score")
    # plt.yticks([0, 0.12, 0.25], ['0', '0.5', '1.0'])
    plt.xlabel("")
    # plt.legend(title="Task Modules", loc='upper right', bbox_to_anchor=(1, 1))
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(save_path_violin, dpi=400, bbox_inches='tight')
    plt.show()

# def plot_and_save_figure(dataset, dataset_0, dataset_1, dataset_2, save_path_bar="cifar10_bar_plot.png", save_path_violin="cifar10_violin_plot.png"):
#     custom_palette = {"VGG19_p00": "#E24A33", "VGG19_p50": "#348ABD", "VGG19_p90": "#988ED5"}

    
#     cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#     # Prepare data for bar plot
#     sample_classes = [dataset.cifar10[i][1] for i in range(len(dataset))]
#     complexity_labels = dataset.complexity_labels
#     complexity_scores = dataset.complexity_scores

#     df_bar = pd.DataFrame({
#         "class": [cifar10_classes[i] for i in sample_classes],
#         "module": ["VGG19_p00" if x == 0 else "VGG19_p50" if x == 1 else "VGG19_p90" for x in complexity_labels],
#         "complexity_score": complexity_scores
#     })

#     count_data = df_bar.groupby(["class", "module"]).size().reset_index(name="count")

#     # Plot bar graph
#     plt.figure(figsize=(10, 5))
#     sns.barplot(x="class", y="count", hue="module", data=count_data)
#     plt.ylabel("# Samples")
#     plt.xlabel("")
#     plt.xticks(rotation=45)
#     plt.legend(title="Modules", loc='upper right', bbox_to_anchor=(1, 1))
#     plt.tight_layout()
#     plt.savefig(save_path_bar, dpi=300)
#     plt.show()

#     # Prepare data for violin plot
#     sample_classes = [dataset_0.cifar10[i][1] for i in range(len(dataset_0))] + [dataset_1.cifar10[i][1] for i in range(len(dataset_1))] + [dataset_2.cifar10[i][1] for i in range(len(dataset_2))]
#     complexity_labels = dataset_0.complexity_labels + dataset_1.complexity_labels + dataset_2.complexity_labels

#     complexity_scores = dataset_0.complexity_scores + dataset_1.complexity_scores + dataset_2.complexity_scores
#     print(complexity_scores[:10])

#     df_violin = pd.DataFrame({
#         "class": [cifar10_classes[i] for i in sample_classes],
#         "module": ["VGG19_p00" if x == 0 else "VGG19_p50" if x == 1 else "VGG19_p90" for x in complexity_labels],
#         "complexity_score": complexity_scores
#     })

#     # Plot violin graph
#     plt.figure(figsize=(10, 5))
#     sns.violinplot(x="class", y="complexity_score", hue="module", data=df_violin, dodge=True, width=1.2, linewidth=1, split=False)
#     plt.ylabel("Complexity Score")
#     plt.xlabel("CIFAR-10 Classes")
#     plt.xticks(rotation=45)
#     plt.legend(title="Modules", loc='upper right', bbox_to_anchor=(1, 1))
#     plt.tight_layout()
#     plt.savefig(save_path_violin, dpi=300)
#     plt.show()



# def plot_and_save_figure(dataset, dataset_0, dataset_1, dataset_2, save_path="cifar10_complexity_plot.png"):
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
    
#     sns.barplot(ax=axes[0], x="class", y="count", hue="module", data=count_data, palette="Pastel1")
#     axes[0].set_ylabel("# Samples")
#     axes[0].set_xlabel("")
#     axes[0].legend(title="Modules", loc='upper right', bbox_to_anchor=(1.1, 1))
    
#     sample_classes = [dataset_0.cifar10[i][1] for i in range(len(dataset_0))] + [dataset_1.cifar10[i][1] for i in range(len(dataset_1))]
#     complexity_labels = dataset_0.complexity_labels + dataset_1.complexity_labels
#     complexity_scores = dataset_0.complexity_scores + dataset_1.complexity_scores
    
#     df = pd.DataFrame({
#         "class": [cifar10_classes[i] for i in sample_classes],
#         "module": ["Large Module" if x == 0 else "Small Module" for x in complexity_labels],
#         "complexity_score": complexity_scores
#     })
    
#     sns.violinplot(ax=axes[1], x="class", y="complexity_score", hue="module", data=df, dodge=True, width=1.2, linewidth=1, palette="Pastel1", split=False)
#     axes[1].set_ylabel("Complexity Score")
#     axes[1].set_xlabel("CIFAR-10 Classes")
    
#     # Set x-axis labels and rotation for both plots
#     axes[0].set_xticklabels(cifar10_classes, rotation=45)
#     axes[1].set_xticklabels(cifar10_classes, rotation=45)
    
#     # Adjust layout and ensure both plots share the same legend
#     axes[0].legend(title="Modules", loc='upper right', bbox_to_anchor=(1.1, 1))
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.show()


args = get_args()

save_dir = 'figures/train'
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

test_dataset_0 = CIFAR10WithSampler(
        args=args,
        model=task_model,
        transform=test_transform,
        train=False,
        module=0
    )
test_dataset_1 = CIFAR10WithSampler(
        args=args,
        model=task_model,
        transform=test_transform,
        train=False,
        module=1
    )
test_dataset_2 = CIFAR10WithSampler(
        args=args,
        model=task_model,
        transform=test_transform,
        train=False,
        module=2
    )
test_dataset = CIFAR10WithSampler_(
    args=args,
    model=task_model,
    transform=test_transform,
    train=False
)

plot_and_save_figure(test_dataset, test_dataset_0, test_dataset_1, test_dataset_2)