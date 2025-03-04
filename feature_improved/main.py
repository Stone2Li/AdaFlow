import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import yaml
import pickle
from scipy.stats import mannwhitneyu
from scipy.stats import wasserstein_distance
from Flow_Matching import Flow_Matching
from train_model import train_flow_matching
def calculate_featrue_score(data,label):
    data = data.squeeze(2)
    health_data=data[label==0].cpu().numpy()
    patient_data=data[label==1].cpu().numpy()
    p_values = []
    for i in range(health_data.shape[1]):
        group1_data = health_data[:, i]
        group2_data = patient_data[:, i]
        stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        p_values.append(p_value)
    p_values = np.array(p_values)
    return torch.tensor(p_values)

#采用百分比版本
"""def select_top_feature(diff_scores, i, threshold = 0.5):
    total_sum = diff_scores.sum()
    target_sum = total_sum * threshold
    sorted_scores, indices = torch.sort(diff_scores)
    cumulative_sum = torch.cumsum(sorted_scores, dim=0)
    selected_idx = torch.where(cumulative_sum <= target_sum)[0]
    top_indices = indices[:len(selected_idx) + 1]  # Include the feature that crosses the threshold    
    return top_indices, len(top_indices)"""
#采用固定阈值版
def select_top_feature(diff_scores, i, threshold = 0.001):
    if threshold - i * 0.001 > 0.001:
        threshold = threshold - i * 0.001
    else: threshold = 0.001
    top_indices = torch.where(diff_scores < threshold)[0]  
    return top_indices, len(top_indices)

def generate_samples(config, i, iteration=50, n_samples=64, seq_len=1000, input_dim=1):
    hidden_dim = config.get('hidden_dim', 128)
    num_heads = config.get('num_heads', 8)
    num_layers = config.get('num_layers', 6)
    save_path = config.get('save_path', './checkpoints')
    save_path = os.path.join(save_path, 'checkpoints')
    model = Flow_Matching(seq_len, hidden_dim=hidden_dim,num_heads=num_heads,num_layers=num_layers)
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
            p_delta =model(p_new, t, labels, seq_len)*dt
            p_new = p_new + p_delta
        return p_new, labels
def compute_wd(old_data,new_data):
    """计算wasserstein距离"""
    old_data = old_data.cpu().numpy()
    new_data = new_data.cpu().numpy()
    seq_len = old_data.shape[1]
    wd_scores = []
    for pos in range(seq_len):
        old_pos = old_data[:, pos, 0]
        new_pos = new_data[:, pos, 0]
        wd = wasserstein_distance(old_pos, new_pos)
        wd_scores.append(wd)
    return np.mean(wd_scores)
def apply_flow_matching(config,data,label,iterations):
    p_0=data.to('cuda:0')  #initial data
    label_0=label.to('cuda:0')
    p = p_0.clone()
    p_out = p_0.clone()
    label_out = label_0.clone()
    threshold = config.get('threshold', 0.25)
    label = label_0.clone()
    all_feature_scores=[]
    all_top_features=[]
    h = 0
    #存储wasserstein距离
    all_wd_0=[]
    all_wd_1=[]
    for i in range(iterations): #样本生成的迭代次数
        p = (p - p.min(dim=0, keepdim=True)[0]) / (p.max(dim=0, keepdim=True)[0] - p.min(dim=0, keepdim=True)[0])
        p_out = (p_out - p_out.min(dim=0, keepdim=True)[0]) / (p_out.max(dim=0, keepdim=True)[0] - p_out.min(dim=0, keepdim=True)[0])
        feature_scores=calculate_featrue_score(p, label)
        top_features,top_n=select_top_feature(feature_scores, i, threshold=threshold) # 选出大于threshold的特征
        if (p.shape[1] == top_n and p.shape[1] <= 100)  or top_n < 2:
            print(f"Break on {i+1}th Iteration, we need {p.shape[1]} dimensions!")
            break
        if p.shape[1] == top_n :
            h += 1
        else : h = 0
        if h == 5:
            print(f'We need {p.shape[1]} dimensions , appear {h} times')
            break
        if top_n == 0:
            print(f"Can't select feature less than {threshold} at {i+1}th Iteration, {p.shape[1]} features we have now!")
            break
        #改为保存所有基因的diffrence
        all_feature_scores.append(feature_scores.cpu().numpy()) 
        all_top_features.append(top_features) #保存前50个特征索引
        p = p[:, top_features]
        p_out = p_out[:, top_features]
        train_flow_matching(config,p,label,i,top_n,iterations)
        p_new,label_new=generate_samples(config, i, seq_len=top_n, iteration=50, n_samples=64) #生成新的样本和标签
        p_out = torch.cat([p_out, p_new], dim=0)
        label_out = torch.cat([label_out, label_new], dim=0)
        # 抽取 p_new 的 1/2
        p_new_health = p_new[:16]
        p_new_disease = p_new[32:48]
        p_new_sampled = torch.cat([p_new_health, p_new_disease], dim=0) # 抽取样本
        p = torch.cat([p, p_new_sampled], dim=0)
        label_health = label_new[:16]
        label_disease = label_new[32:48]
        label_new_sampled = torch.cat([label_health,label_disease], dim=0)
        label = torch.cat([label,label_new_sampled], dim=0)
        #计算wasserstein距离
        wd_0=compute_wd(p[:64][label_0==0],p_new[label_new==0])
        wd_1=compute_wd(p[:64][label_0==1],p_new[label_new==1])
        all_wd_0.append(wd_0)
        all_wd_1.append(wd_1)
        # p=torch.cat([p,p_new],dim=0)
        # label = torch.cat([label,label_new])
        print(f"[{i+1}/{iterations}] Wasserstein_0:{wd_0:.4f} Wasserstein_1:{wd_1:.4f}")
        print(f'Saved feature selected:{p.shape[1]}')
    return p_out, label_out, all_feature_scores, all_top_features, all_wd_0, all_wd_1

if __name__=="__main__":
    config = "feature_improved/train_config.yaml"
    config = yaml.load(open(config, 'rb'), Loader=yaml.FullLoader)
    iteration = 100
    df=pd.read_csv("top_1000_genes_PCA_data.txt",delim_whitespace=True,header=None)
    columns=df.iloc[:,0]
    data=pd.DataFrame(df.iloc[1:,1:].values.T,columns=list(columns[1:]))
    label=list(df.iloc[0,1:].apply(lambda x:1 if "Y" in x else 0 if 'Z' in x else -1))
    # 转换数据为Tensor
    data = torch.tensor(data.values.astype(np.float32), dtype=torch.float32)
    data = data.unsqueeze(2)
    label = torch.tensor(label, dtype=torch.long)
    p, label, all_feature_scores,all_top_features,all_wd_0,all_wd_1 = apply_flow_matching(config, data, label, iteration)
    results = {
        'samples': p,
        'labels': label,
        'all_feature_scores': all_feature_scores,
        'all_top_features': all_top_features,
        'all_wd_0': all_wd_0,
        'all_wd_1': all_wd_1
    }
    save_path = config.get('save_path', './results/wrong')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, 'results.pkl')
    with open(save_path, 'wb') as f:
        f.seek(0)
        f.truncate()
        pickle.dump(results, f)