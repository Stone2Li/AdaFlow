import matplotlib.pyplot as plt
import numpy as np
import pickle
def track_gene_scores(all_feature_scores, all_top_features, num_genes=1000, iterations=50):
    """
    跟踪每个基因在多次迭代中的得分变化，并填充缺失的得分
    """
    # 初始化字典存储每个基因的得分列表
    gene_scores = {i: [] for i in range(num_genes)}
    
    # 遍历每次迭代
    for iteration in range(iterations):
        top_genes = all_top_features[iteration]  # 当前迭代选出的最显著基因的索引
        feature_scores = all_feature_scores[iteration]  # 当前迭代中基因的得分
        
        # 为当前迭代未选中的基因填充NaN
        for gene_idx in range(num_genes):
            if gene_idx in top_genes:
                index = np.where(top_genes == gene_idx)[0]
                score = feature_scores[index]  # 获取当前基因的得分
            else:
                score = np.nan  # 没有被选中的基因得分填充为NaN
            gene_scores[gene_idx].append(score)
    
    return gene_scores

def visualize_gene_scores(gene_scores, iterations=50, top_n=10):
    """
    可视化前 top_n 个基因的得分变化
    """
    plt.figure(figsize=(15, 10))

    # 选择最显著的 top_n 个基因（按顺序）
    for gene_idx in range(top_n):
        scores = gene_scores[gene_idx]
        plt.plot(range(iterations), scores, label=f'Gene {gene_idx+1}')

    plt.xlabel('Iterations')
    plt.ylabel('Gene Score')
    plt.title(f'Gene Score Changes Over {iterations} Iterations')
    plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), shadow=True)
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    # 读取保存的字典文件
    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)
    iteration=10
    # 提取特征分数和前特征索引
    all_feature_scores = results['all_feature_scores']
    all_feature_scores = np.squeeze(all_feature_scores)
    all_top_features = results['all_top_features']
    print(np.shape(all_feature_scores))
    print(np.shape(all_top_features))
    gene_scores = track_gene_scores(all_feature_scores, all_top_features, num_genes=1000, iterations=iteration)
    # 保存 gene_scores 到 pickle 文件
    with open('gene_scores.pkl', 'wb') as f:
        pickle.dump(gene_scores, f)
    # 统计每个基因的nan数量
    nan_counts = {}
    for gene_idx, scores in gene_scores.items():
        nan_count = sum(np.isnan(score) for score in scores)
        nan_counts[gene_idx] = nan_count
    
    # 按nan数量降序排序
    sorted_genes = sorted(nan_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("基因索引及其对应的nan数量(降序排列):")
    for gene_idx, count in sorted_genes:
        print(f"基因 {gene_idx}: {count} 个nan值")