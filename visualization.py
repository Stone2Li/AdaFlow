import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import pickle
from scipy.stats import mannwhitneyu
from sklearn.manifold import SpectralEmbedding
import yaml
import os
def plot_feature_scores(all_feature_scores,save_path):
    """画出迭代过程中特征得分的箱形图"""
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=all_feature_scores.T)
    plt.title("Feature Scores Distribution Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Feature Score")
    plt.xticks(range(10), range(1, 11))
    plt.savefig(f"{save_path}/Feature Scores Distribution Across Iterations.png")

def visualize_data_distribution(p_0, p_new, label_0, label_new, iterations, save_path):
    """用PCA可视化生成样本呢和初始样本的散点图"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    fig.suptitle("Data Distribution Across Iterations", fontsize=16)
    pca = PCA(n_components=2)
    p_0_flat = p_0.cpu().numpy().reshape(p_0.shape[0], -1)  # [n_samples, seq_len * input_dim]
    p_flat = p_new.cpu().numpy().reshape(p_new.shape[0], -1)
    label_0_np = label_0.cpu().numpy()
    label_np = label_new.cpu().numpy()
    health_data = p_0_flat[label_0_np==0]
    disease_data = p_0_flat[label_0_np==1]
    p_values = []
    for i in range(p_0_flat.shape[1]):
        stat, p_value = mannwhitneyu(health_data[:, i], disease_data[:, i], alternative='two-sided')
        p_values.append(p_value)
    top_50_p_values_indices = np.argsort(p_values)[:50]
    dm = SpectralEmbedding(n_components=2, affinity='nearest_neighbors', random_state=42)#cosine/rbf
    top_data = p_0_flat[:, top_50_p_values_indices]
    embedding_0 = dm.fit_transform(top_data)
    for i, ax in enumerate(axes.flat):
        if i < iterations:
            start_idx = i * 64
            end_idx = (i + 1) * 64
            p_new_flat = p_flat[start_idx:end_idx]  # [64, seq_len * input_dim]
            label_new_np = label_np[start_idx:end_idx]  # [64]
            p_new_flat = p_new_flat[:, top_50_p_values_indices]
            embedding_1 = dm.fit_transform(p_new_flat)
            ax.scatter(embedding_0[label_0_np==0, 0], embedding_0[label_0_np==0, 1], c='blue', label='Old Health', alpha=0.5)
            ax.scatter(embedding_0[label_0_np==1, 0], embedding_0[label_0_np==1, 1], c='red', label='Old Disease', alpha=0.5)
            ax.scatter(embedding_1[label_new_np==0, 0], embedding_1[label_new_np==0, 1], c='lightblue', label='New Health', alpha=0.5)
            ax.scatter(embedding_1[label_new_np==1, 0], embedding_1[label_new_np==1, 1], c='pink', label='New Disease', alpha=0.5)
            ax.set_title(f"Iter {i+1}")
            if i == 0:
                ax.legend()
            if i % 5 == 0:
                ax.set_ylabel("Component 2")
            if i >= 5:
                ax.set_xlabel("Component 1")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{save_path}/Data Distribution across Iterations.png")

def plot_wasserstein_distance(all_wd_0, all_wd_1, save_path):
    """Plot Wasserstein Distance across Iterations"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(all_wd_0) + 1), all_wd_0, label='Health Group', marker='o')
    plt.plot(range(1, len(all_wd_1) + 1), all_wd_1, label='Disease Group', marker='o')
    plt.title("Wasserstein Distance Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Wasserstein Distance")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/Wasserstein Distance Across Iterations.png")

def plot_top_features_stability(all_top_features, save_path):
    """Plot the counts of genes across Iterations"""
    feature_counts = np.zeros(1000)  # 假设有 1000 个基因
    for top_features in all_top_features:
        feature_counts[top_features] += 1
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(1000), feature_counts)
    plt.title("Top Features Stability Across Iterations")
    plt.xlabel("Feature Index")
    plt.ylabel("Frequency in Top 50")
    plt.savefig(f"{save_path}/Top Features Stability Across Iterations.png")

def plot_generated_feature_scores(p_0, p_new, label_0, label_new, top_features, iterations, save_path):
    """Plot top 50 genes scores distribution across Iterations"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    fig.suptitle("Generated Feature Scores Distribution Across Iterations", fontsize=16)
    p_0_np = p_0.cpu().numpy().squeeze()
    p_np = p_new.cpu().numpy().squeeze()
    label_0_np = label_0.cpu().numpy()
    label_np = label_new.cpu().numpy()
    for i, ax in enumerate(axes.flat):
        if i < iterations:
            start_idx = i * 64
            end_idx = (i + 1) * 64
            p_new_np = p_np[start_idx:end_idx]  # [64, seq_len]
            label_new_np = label_np[start_idx:end_idx]  # [64]
            top_features = all_top_features[i]  # 当前迭代的 top 50 特征
            
            top_scores_old = p_0_np[:, top_features].mean(axis=1)  # [64]
            top_scores_new = p_new_np[:, top_features].mean(axis=1)  # [64]
            
            data = [top_scores_old[label_0_np==0], top_scores_old[label_0_np==1], 
                    top_scores_new[label_new_np==0], top_scores_new[label_new_np==1]]
            sns.violinplot(data=data, ax=ax)
            ax.set_xticks([0, 1, 2, 3], ['OH', 'OD', 'NH', 'ND'])  # Old Health, Old Disease, New Health, New Disease
            ax.set_title(f"Iter {i+1}")
            if i % 5 == 0:
                ax.set_ylabel("Mean Score of Top Features")
            else:
                ax.set_ylabel("")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{save_path}/Feature Scores Distribution for Top Features.png")

if __name__=='__main__':
    # 读取保存的字典文件
    with open('./results/origin/results.pkl', 'rb') as f:
        results = pickle.load(f)
    config = "./train_config.yaml"
    config = yaml.load(open(config, 'rb'), Loader=yaml.FullLoader)
    save_path = config.get('save_path', './results/wrong')
    save_path = os.path.join(save_path, 'pictures')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    iterations=10
    n_samples = 64
    # 提取特征分数和前特征索引
    all_feature_scores = results['all_feature_scores']
    all_feature_scores = np.squeeze(all_feature_scores)
    all_top_features = results['all_top_features']
    all_wd_0 = results['all_wd_0']
    all_wd_1 = results['all_wd_1']
    samples = results['samples']
    labels = results['labels']
    samples = samples.squeeze(2)
    plot_feature_scores(all_feature_scores, save_path=save_path)
    plot_wasserstein_distance(all_wd_0, all_wd_1, save_path=save_path)
    plot_top_features_stability(all_top_features,save_path=save_path)
    visualize_data_distribution(samples[:n_samples, :], samples[n_samples:, :], labels[:n_samples], labels[n_samples:], iterations=iterations,save_path=save_path)
    plot_generated_feature_scores(samples[:n_samples, :],samples[n_samples:, :], labels[:n_samples],labels[n_samples:], top_features=all_top_features, iterations=iterations,save_path=save_path)
