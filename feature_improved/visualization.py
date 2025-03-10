import matplotlib.pyplot as plt
import torch
import pandas as pd
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
    features = all_feature_scores[:]
    sns.boxplot(data=features)
    plt.title("Feature Scores Distribution Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Feature Score")
    plt.xticks(range(10), range(1, 11))
    plt.savefig(f"{save_path}/Feature Scores Distribution Across Iterations.png")

def visualize_data_distribution(p_0, p_new, label_0, label_new, iterations, save_path):
    """可视化生成样本呢和初始样本的散点图"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    fig.suptitle("Data Distribution Across Iterations", fontsize=16)
    p_0_flat = p_0.cpu().numpy().reshape(p_0.shape[0], -1)  # [n_samples, seq_len * input_dim]
    p_flat = p_new.cpu().numpy().reshape(p_new.shape[0], -1)
    label_0_np = label_0.cpu().numpy()
    label_np = label_new.cpu().numpy()
    dm = PCA(n_components=2, whiten=True, random_state=42)  # 使用PCA进行降维，保留2个主成分并进行白化    
    # dm = SpectralEmbedding(n_components=2, affinity='nearest_neighbors', random_state=42)#cosine/rbf
    embedding_0 = dm.fit_transform(p_0_flat)
    for i, ax in enumerate(axes.flat):
        if i < iterations:
            start_idx = i * 64
            end_idx = (i + 1) * 64
            p_new_flat = p_flat[start_idx:end_idx]  # [64, seq_len * input_dim]
            label_new_np = label_np[start_idx:end_idx]  # [64]
            embedding_1 = dm.transform(p_new_flat)
            ax.scatter(embedding_0[label_0_np==0, 0], embedding_0[label_0_np==0, 1], c='blue', label='Old Health', alpha=0.5)
            ax.scatter(embedding_0[label_0_np==1, 0], embedding_0[label_0_np==1, 1], c='red', label='Old Disease', alpha=0.5)
            ax.scatter(embedding_1[label_new_np==0, 0], embedding_1[label_new_np==0, 1], c='green', label='New Health', alpha=1)
            ax.scatter(embedding_1[label_new_np==1, 0], embedding_1[label_new_np==1, 1], c='yellow', label='New Disease', alpha=1)
            ax.set_title(f"Iter {i+1}")
            if i == 0:
                ax.legend()
            if i % 5 == 0:
                ax.set_ylabel("Component 2")
            if i >= 5:
                ax.set_xlabel("Component 1")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{save_path}/Data Distribution across Iterations.png")

def visualize_data_distribution_top_dimension(p_0, p_new, label_0, label_new, all_top_features, all_feature_scores, iterations, save_path):
    """可视化生成样本呢和初始样本的散点图"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    fig.suptitle("Data Distribution Across Iterations", fontsize=16)
    p_0_flat = p_0.cpu().numpy().reshape(p_0.shape[0], -1)  # [n_samples, seq_len * input_dim]
    p_flat = p_new.cpu().numpy().reshape(p_new.shape[0], -1)
    label_0_np = label_0.cpu().numpy()
    label_np = label_new.cpu().numpy()
    embedding_0 = p_0_flat
    # 筛选all_feature_score[-1]中最小两个数的index
    last_feature_scores = all_feature_scores[-1]
    min_two_indices = np.argsort(last_feature_scores)[:2]
    
    # 提取p_0和p_new中对应index的数据
    embedding_0 = p_0_flat[:, min_two_indices]

    for i, ax in enumerate(axes.flat):
        if i < iterations:
            start_idx = i * 64
            end_idx = (i + 1) * 64
            p_new_flat = p_flat[start_idx:end_idx]  # [64, seq_len * input_dim]
            label_new_np = label_np[start_idx:end_idx]  # [64]
            embedding_1 = p_new_flat[:, min_two_indices]

            ax.scatter(embedding_0[label_0_np==0, 1], embedding_0[label_0_np==0, 0], c='blue', label='Old Health', alpha=0.5)
            ax.scatter(embedding_0[label_0_np==1, 1], embedding_0[label_0_np==1, 0], c='red', label='Old Disease', alpha=0.5)
            ax.scatter(embedding_1[label_new_np==0, 1], embedding_1[label_new_np==0, 0], c='green', label='New Health', alpha=0.5)
            ax.scatter(embedding_1[label_new_np==1, 1], embedding_1[label_new_np==1, 0], c='yellow', label='New Disease', alpha=0.5)
            ax.set_title(f"Iter {i+1}")
            if i == 0:
                ax.legend()
            if i % 5 == 0:
                ax.set_ylabel("Component 2")
            if i >= 5:
                ax.set_xlabel("Component 1")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{save_path}/Data Distribution across Iterations_top_feature.png")

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

def plot_top_features_stability(all_top_features, columns, save_path):
    """Plot the counts of genes across Iterations"""
    columns = all_top_features[0]
    feature_counts = pd.DataFrame(np.zeros((1,len(columns))), columns=columns)  # 假设有 1000 个基因
    for top_features in all_top_features:
        feature_counts[top_features] += 1
    
    plt.figure(figsize=(12, 6))
    # 使用列名作为横坐标，纵轴为值
    plt.bar(columns, feature_counts.sum(axis=0))  # 计算每列的总和
    plt.title("Top Features Stability Across Iterations")
    plt.xlabel("Feature Index")
    plt.ylabel("Frequency in Across Iterations")
    plt.xticks(rotation=90, fontsize=4)  # 旋转横坐标标签以便更好地显示
    plt.savefig(f"{save_path}/Top Features Stability Across Iterations.png")
    return feature_counts

def plot_diff_curve(p, labels, iterations):
    """Plot diff_mean across iterations"""
    diff = []
    for i in range(iterations+1):
        start_index = i * 64
        end_index = (i + 1) * 64
        data = p[start_index : end_index]
        label = labels[start_index : end_index]
        data_health = data[label == 0]
        data_disease = data[label == 1]
        diff_mean = torch.mean(torch.abs(torch.mean(data_health, dim=0) - torch.mean(data_disease, dim=0)))
        diff.append(diff_mean.cpu().numpy())
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(diff) + 1), diff, marker='o')
    plt.title("Averaged Mean Distance Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Averaged Mean Distance")
    plt.grid(True)
    plt.savefig(f"{save_path}/Averaged Mean Distance Across Iterations.png")
 
if __name__=='__main__':
    config = "feature_improved/train_config.yaml"
    config = yaml.load(open(config, 'rb'), Loader=yaml.FullLoader)
    save_path = config.get('save_path', './results/wrong')
    # 读取保存的字典文件
    with open(f'{save_path}/results.pkl', 'rb') as f:
        results = pickle.load(f)
    save_path = os.path.join(save_path, 'pictures')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    iterations= 6
    n_samples = 64
    # 提取特征分数和前特征索引
    all_feature_scores = results['all_feature_scores']
    #all_feature_scores = np.squeeze(all_feature_scores)
    all_top_features = results['all_top_features']
    all_wd_0 = results['all_wd_0']
    all_wd_1 = results['all_wd_1']
    samples = results['samples']
    columns = results['columns']
    labels = results['labels']
    samples = samples.squeeze(2)
    plot_diff_curve(samples, labels, iterations=iterations)
    plot_feature_scores(all_feature_scores, save_path=save_path)
    plot_wasserstein_distance(all_wd_0, all_wd_1, save_path=save_path)
    feature_counts = plot_top_features_stability(all_top_features, columns, save_path=save_path)
    visualize_data_distribution(samples[:n_samples, :], samples[n_samples:, :], labels[:n_samples], labels[n_samples:], iterations=iterations,save_path=save_path)
    visualize_data_distribution_top_dimension(samples[:n_samples, :], samples[n_samples:, :], labels[:n_samples], labels[n_samples:], all_top_features, all_feature_scores,
                                              iterations, save_path)
