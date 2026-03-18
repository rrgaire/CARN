import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from fpdf import FPDF

from fvcore.nn import FlopCountAnalysis


from main import *
from models import *

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
                    entropy = self.compute_entropy(logits).item() / np.log(10)
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


def plot_pie_chart(labels, save_dir):
    plt.figure(figsize=(6,6))
    label_counts = Counter(labels)
    plt.pie(label_counts.values(), labels=label_counts.keys(), autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title("Distribution of Assigned Complexity Labels")
    plt.savefig(f"{save_dir}/pie_chart.png")
    plt.close()

def plot_violin_plot(classes, complexity_scores, save_dir):
    plt.figure(figsize=(10,6))
    sns.violinplot(x=classes, y=complexity_scores)
    plt.xlabel("CIFAR-10 Classes")
    plt.ylabel("Complexity Score")
    plt.title("Complexity Score Distribution per Class")
    plt.xticks(rotation=45)
    plt.savefig(f"{save_dir}/violin_plot.png")
    plt.close()

def plot_tsne_pca(features, labels, save_dir):
    features = np.array([f.flatten() for f in features])
    features = StandardScaler().fit_transform(features)
    pca = PCA(n_components=50).fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(pca)
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=labels, palette='tab2', alpha=0.7)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.title("t-SNE Projection of CIFAR-10 Samples")
    plt.legend(title="Task Module")
    plt.savefig(f"{save_dir}/tsne_pca_plot.png")
    plt.close()

def plot_3d_scatter(entropies, confidences, labels, save_dir):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(entropies, confidences, c=labels, cmap='tab2', alpha=0.7)
    ax.set_xlabel("Entropy")
    ax.set_ylabel("1 - Confidence")
    ax.set_zlabel("Complexity Score")
    plt.title("3D Scatter Plot of Sample Complexity")
    plt.colorbar(sc, label="Task Module")
    plt.savefig(f"{save_dir}/3d_scatter_plot.png")
    plt.close()

def generate_pdf(features, complexity_scores, entropies, confidences, save_dir):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    os.makedirs("figures_train/pdf_samples", exist_ok=True)
    
    sample_indices = np.random.choice(len(features), 20, replace=False)
    
    for i, idx in enumerate(sample_indices):
        pdf.add_page()
        
        # Save and insert image
        plt.figure(figsize=(3,3))
        plt.imshow(np.mean(features[idx], axis=0), cmap='gray')
        img_path = f"figures_train/pdf_samples/sample_{i}.png"
        plt.axis('off')
        plt.savefig(img_path)
        plt.close()
        pdf.image(img_path, x=10, y=20, w=60)
        
        # Save and insert stacked bar graph
        plt.figure(figsize=(3,3))
        components = [1 - confidences[idx], entropies[idx]]
        labels_bar = ['1-Confidence', 'Entropy']
        plt.bar(labels_bar, components, color=['blue', 'orange'])
        plt.ylim(0, 1)
        plt.title("Complexity Score Breakdown")
        bar_path = f"figures_train/pdf_samples/bar_{i}.png"
        plt.savefig(bar_path)
        plt.close()
        pdf.image(bar_path, x=80, y=20, w=60)
    
    pdf.output(f"{save_dir}/sample_analysis.pdf")


if __name__ == '__main__':

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

    # val_dataset = CIFAR10WithSampler(
    #     args=args,
    #     model=task_model,
    #     transform=test_transform, 
    #     train=True
    # )


    # all_indices = list(np.arange(len(train_dataset)))
    # val_size = int(args.val_split * len(train_dataset))
    # val_indices = random.sample(all_indices, val_size)
    # train_indices = np.setdiff1d(all_indices, val_indices)

    # train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    # val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    # t_sampler = torch.utils.data.WeightedRandomSampler(
    # weights=train_dataset.weights,
    # num_samples=len(train_dataset.weights),
    # replacement=True 
    
    # )
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_subset,
    #     batch_size=args.batch_size,
    #     drop_last=True
    # )

    # val_dataloader = torch.utils.data.DataLoader(
    #     val_subset,
    #     batch_size=args.batch_size,
    #     shuffle=False, 
    #     drop_last=True
    # )


    # Compute dataset-wide statistics and visualize
    entropies = train_dataset.entropies
    confidences = train_dataset.confidences
    energy_costs = [train_dataset.energy_costs[label] for label in train_dataset.complexity_labels]
    labels = train_dataset.complexity_labels

    # Generate and save the visualizations
    plot_pie_chart(labels, save_dir)
    plot_violin_plot([train_dataset.cifar10.targets[i] for i in range(len(train_dataset))], labels, save_dir)
    plot_tsne_pca(train_dataset.features, train_dataset.complexity_labels, save_dir)
    plot_3d_scatter(entropies, confidences, labels, save_dir)
    generate_pdf(features=train_dataset.features, complexity_scores=train_dataset.complex)

    print("Visualization plots saved in the 'figures' directory.")