import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import SpectralEmbedding
import seaborn as sns
import matplotlib.pyplot as plt


data=pd.read_csv("top_1000_genes_PCA_data.txt",delim_whitespace=True, header=None)#代码适配的文件没有第一行的label
genes=data[0]
genes_data = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values  # 强制转换为数值类型
scaler = StandardScaler()
gene_standard = np.apply_along_axis(lambda x: (x - np.mean(x)) / np.std(x), axis=1, arr=genes_data)
groups = np.array(['group1'] * 32 + ['group2'] * 32)
#Mann-Whitney U 双边检验
p_values = []
for i in range(gene_standard.shape[0]):
    group1_data = gene_standard[i, :32]
    group2_data = gene_standard[i, 32:]
    stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
    p_values.append(p_value)

p_values_df = pd.DataFrame({'gene': genes, 'p_value': p_values})
top_50_genes = p_values_df.sort_values(by='p_value').head(50)
#top_50_genes.to_csv('top_50_genes_DP.csv', index=False)

#diffusion map
top_50_data = gene_standard[top_50_genes.index, :]
embedding = SpectralEmbedding(n_components=2, affinity='nearest_neighbors', random_state=42)#cosine/rbf
diffusion_map_result = embedding.fit_transform(top_50_data.T)
diffusion_map_df = pd.DataFrame(diffusion_map_result, columns=['DM1', 'DM2'])
diffusion_map_df['Group'] = groups

plt.figure(figsize=(8, 6))
sns.scatterplot(x='DM1', y='DM2', hue='Group', data=diffusion_map_df, palette="Set1", s=100)
plt.title("Top 50 差异基因的 Diffusion Map 降维可视化")
plt.xlabel("Diffusion Map 1 ")
plt.ylabel("Diffusion Map 2 ")
plt.legend(title='组别')
plt.savefig('top_50_genes_diffusion_map.png')
plt.show()
