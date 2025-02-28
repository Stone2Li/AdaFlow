import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import yaml
import pickle
from scipy.stats import wasserstein_distance
from Flow_Matching import Flow_Matching
from train_model import train_flow_matching
from scipy.stats import pearsonr
from datafold.dynfold import DiffusionMaps
def diffusion_map(data, n_components=50, alpha=1.0):
    # 添加数据校验
    assert not np.isnan(data).any(), "Input data contains NaN values"
    assert not np.isinf(data).any(), "Input data contains infinite values"
    dm = DiffusionMaps(n_eigenpairs=n_components, alpha=alpha, dist_kwargs=dict(
            n_harmonics=n_components+5,  # NCV = NEV + 5
            tol=1e-6,  # 降低收敛阈值
            maxiter=5000  # 增加最大迭代次数
            ))
    embedding = dm.fit_transform(data)
    eigenvalues = dm.eigenvalues_
    eigenvectors = dm.eigenvectors_
    return embedding, eigenvalues, eigenvectors
def calculate_featrue_scores(original_data, eigenvectors):
    n_features = original_data.shape[1]  
    n_components = eigenvectors.shape[1]    
    feature_scores = np.zeros(n_features)
    eigenvectors = eigenvectors[:original_data.shape[0], :]
    for feature_idx in range(n_features):
        feature_data = original_data[:, feature_idx]  # 第 i 个基因的数据
        correlations = []
        for component_idx in range(n_components):
            component_data = eigenvectors[:, component_idx]  # 第 j 个降维维度的数据
            corr, _ = pearsonr(feature_data, component_data)  # 计算皮尔逊相关系数
            correlations.append(np.abs(corr))  # 取绝对值，捕捉正负相关性
        feature_scores[feature_idx] = np.max(correlations)  # 对所有维度的相关性取平均

    return feature_scores
def select_top_feature(feature_scores, top_n=50):
    top_indices = np.argsort(feature_scores)[:top_n]
    return top_indices

def select_top_k_eigenvectors(eigenvalues, threshold=0.9):
    cumulative_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    k = np.argmax(cumulative_var >= threshold) + 1
    return k

def generate_samples(config, i, k, iteration=50, n_samples=64, input_dim=1):
    seq_len = k
    hidden_dim = config.get('hidden_dim', 128)
    num_heads = config.get('num_heads', 8)
    num_layers = config.get('num_layers', 6)
    save_path = config.get('save_path', './checkpoints')
    save_path = os.path.join(save_path, 'checkpoints')
    model = Flow_Matching(seq_len, hidden_dim=hidden_dim,num_heads=num_heads,num_layers=num_layers,output_dim=k)
    checkpoint_path = os.path.join(save_path, f'model_{i+1}th Iteration.pth') #使用的模型参数
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.to(model.device)
    print(f"Loaded model from {checkpoint_path}")
    model.eval()
    # with torch.no_grad():  # 无需梯度，加速，降显存
    with torch.no_grad():
        dt = 1.0/iteration
        t = torch.zeros(n_samples).to(model.device)
        x_0 = torch.randn(n_samples, seq_len, input_dim).to(model.device)
        p_new = x_0.clone().to(model.device)
        labels = torch.cat([torch.ones(32, dtype=torch.long), torch.zeros(32, dtype=torch.long)]).to(model.device)
        for i in range(iteration):
            t += dt
            p_delta =model(p_new, t, labels,seq_len=seq_len)*dt
            p_new = p_new + p_delta
        return p_new, labels
def compute_wd(old_data,new_data):
    """计算wasserstein距离"""
    old_data = old_data.cpu().numpy().flatten()
    new_data = new_data.cpu().numpy().flatten()
    return wasserstein_distance(old_data,new_data)
def apply_flow_matching(config,data,label,iterations,top_num=50):
    p_0=data.to('cuda:0')  #initial data
    label_0=label.to('cuda:0')
    k = config.get('initial_k', 50)
    embedding_tensor = p_0.clone()
    label = label_0
    #对p_0归一化
    p_0_num = p_0.squeeze(2).cpu().numpy()
    mean_0 = np.mean(p_0_num, axis=0)
    std_0 = np.std(p_0_num, axis=0)
    p_0_num = (p_0_num-mean_0) / (std_0 + 1e-8)
    #将p_0也做降维保存用于判断生成效果
    embedding_0,_,_ = diffusion_map(p_0.squeeze(2).cpu().numpy(), n_components=k)
    embedding_tensor_0 = torch.tensor(embedding_0, dtype=torch.float32).to('cuda:0') #降维后的数据   
    embedding_tensor_0 = embedding_tensor_0.unsqueeze(2)
    all_feature_scores=[]
    all_top_features=[]
    #存储wasserstein距离
    all_wd_0=[]
    all_wd_1=[]
    for i in range(iterations): #样本生成的迭代次数
        current_data = embedding_tensor.squeeze(2).cpu().numpy()
        mean = np.mean(current_data, axis=0)
        std = np.std(current_data, axis=0)
        normalized_data = (current_data-mean) / (std + 1e-8)
        # Diffusion Map降维
        embedding, eigenvalues, eigenvectors = diffusion_map(normalized_data, n_components=k) #####这里有问题，可能需要随机的取64个样本点进行diffusionmap
        k_new = select_top_k_eigenvectors(eigenvalues)  # 动态选择主要方向数量
        feature_scores = calculate_featrue_scores(p_0.squeeze(2).cpu().numpy(), eigenvectors)      
        top_features=select_top_feature(feature_scores, top_num) # 选出差异最大的50个特征
        #all_feature_scores.append(feature_scores[top_features].cpu().numpy())
        #改为保存所有基因的diffrence
        all_feature_scores.append(feature_scores) 
        all_top_features.append(top_features) #保存前50个特征索引
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to('cuda:0') #降维后的数据   
        embedding_tensor = embedding_tensor.unsqueeze(2)
        train_flow_matching(config,embedding_tensor,label,i,k,iterations)
        p_new,label_new=generate_samples(config,i,k, iteration=50, n_samples=64) #生成新的样本和标签
        #计算wasserstein距离
        wd_0=compute_wd(embedding_tensor_0[label_0==0],p_new[label_new==0])
        wd_1=compute_wd(embedding_tensor_0[label_0==1],p_new[label_new==1])
        all_wd_0.append(wd_0)
        all_wd_1.append(wd_1)
        embedding_tensor = torch.cat([embedding_tensor,p_new],dim=0)
        label=torch.cat([label,label_new])
        print(f"[{i+1}/{iterations}] Wasserstein_0:{wd_0:.4f} Wasserstein_1:{wd_1:.4f}")
        if k == k_new:
            print(f"Break on {i+1}th Iteration, we need {k} dimensions!")  
            break
        else:
            k = k_new
    return embedding_tensor, label, all_feature_scores, all_top_features, all_wd_0, all_wd_1

if __name__=="__main__":
    config = "./train_config.yaml"
    config = yaml.load(open(config, 'rb'), Loader=yaml.FullLoader)
    iteration = config.get('Iteration', 100)
    df=pd.read_csv("top_1000_genes_PCA_data.txt",delim_whitespace=True,header=None)
    columns=df.iloc[:,0]
    data=pd.DataFrame(df.iloc[1:,1:].values.T,columns=list(columns[1:]))
    label=list(df.iloc[0,1:].apply(lambda x:1 if "Y" in x else 0 if 'Z' in x else -1))
    # 转换数据为Tensor
    data = torch.tensor(data.values.astype(np.float32), dtype=torch.float32)
    data = data.unsqueeze(2)
    label = torch.tensor(label, dtype=torch.long)
    samples,label,all_feature_scores,all_top_features,all_wd_0,all_wd_1 = apply_flow_matching(config, data, label, iteration)
    results = {
        'all_feature_scores': all_feature_scores,
        'all_top_features': all_top_features,
        'samples': samples,
        'label': label,
        'all_wd_0': all_wd_0,
        'all_wd_1': all_wd_1
    }
    save_path = config.get('save_path', './results/wrong')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, 'results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)