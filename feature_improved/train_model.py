import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from torch.optim import Adam
import time
import yaml
from Flow_Matching import Flow_Matching  # 导入模型
import torch.nn.functional as F
class CustomDataset(Dataset):
    """自定义数据集"""
    def __init__(self, data, labels):
        """
        Args:
            data (torch.Tensor): 输入数据（例如 DNA 数据），形状应为 [batch_size, seq_len, input_dim]
            time (torch.Tensor): 时间戳数据
            labels (torch.Tensor): 标签数据
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_flow_matching(config: str, data, labels, i, top_n, iterations=10):
    """训练Flow Matching模型"""
    # 获取配置
    seq_len = top_n
    hidden_dim = config.get('hidden_dim', 128)
    num_heads = config.get('num_heads', 8)
    num_layers = config.get('num_layers', 6)
    epochs = config.get('epochs', 1000)
    base_batch_size = config.get('base_batch_size', 16)
    learning_rate = config.get('learning_rate', 1e-3)
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_save_interval = config.get('checkpoint_save_interval', 1)
    save_path = config.get('save_path', './checkpoints')
    #随着数据的增多增加batch_size
    if (i+1)*base_batch_size <64:
        batch_size = (i+1)*base_batch_size
    else:
        batch_size = 64
    if epochs + i * 10 < 50:
        epochs = epochs + i * 10
    else: 
        epochs = 50
    # 创建数据集和数据加载器
    dataset = CustomDataset(data, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = 1
    output_dim = 1000
    model = Flow_Matching(seq_len, input_dim, hidden_dim, num_heads, num_layers, output_dim).to(device)

    # 损失函数和优化器
    optimizer = Adam(model.parameters(), lr=learning_rate)

    save_path = os.path.join(save_path, 'checkpoints')
    # 创建模型保存目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    loss_list = []
    start_time = time.time()
    # 开始训练
    for epoch in range(epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0

        # 训练一个epoch
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # 将数据移动到设备上
            x_1, labels = inputs.to(device), labels.to(device)
            # 创建flow
            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.size(0)).to(device)
            x_t = t.unsqueeze(1).unsqueeze(2) * x_1 + (1-t.unsqueeze(1).unsqueeze(2)) * x_0

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            v_pred = model(x_t, t, labels, seq_len=seq_len)
            x_new = x_0 + v_pred
            # 计算x_new中label=0和label=1各维度平均值的差距的平均
            mean_diff = torch.mean(torch.abs(torch.mean(x_new[labels==0], dim=0) - torch.mean(x_new[labels==1], dim=0))) #在损失函数中加入两类的区别以防止丢失信息
            # 计算x_new到x_1每个样本距离的平均（欧式距离）
            distance = torch.mean(torch.norm(x_new - x_1, dim=1)) / x_1.shape[1]

            # distance 限制生成的样本质量：若系数设置太低，则会导致筛选的特征主要用于区分生成的新样本，对原始样本区分不大
            # mean_diff: 用于拉开两类样本，在原本样本差距不大的基本上，逐渐拉开样本，用于筛选关键特征
            # Flow_matching_loss : 用于规定样本生成方式，便于之后生成

            loss = torch.exp(1 * F.mse_loss(x_1-x_0, v_pred)+ 2 * distance + 0.0 * mean_diff)
            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 累计损失
            running_loss += loss.item()

        # 输出每个epoch的平均损失
        avg_loss = running_loss / len(train_loader)
        loss_list.append(avg_loss)
        print(f'Iteration[{i+1}/{iterations}], Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

    # 保存模型检查点
    finish_time = time.time()-start_time
    checkpoint_path = os.path.join(save_path, f'model_{i+1}th Iteration.pth')
    save_dict = dict(model=model.state_dict(),
                     optimizer=optimizer.state_dict(),
                     epoch=epoch,
                     loss_list=loss_list,
                     time=finish_time)
    torch.save(save_dict,checkpoint_path)
    print(f'Model checkpoint saved at {checkpoint_path}')

    print('Training Finished')


if __name__ == '__main__':
    # 配置文件路径
    config_path = 'train_config.yaml'
    config = yaml.load(open(config_path, 'rb'), Loader=yaml.FullLoader)
    data = torch.randn(64, 1000, 1)
    labels = torch.randint(0, 2, (1000,))
    train_flow_matching(config_path, data, labels,i=1)
