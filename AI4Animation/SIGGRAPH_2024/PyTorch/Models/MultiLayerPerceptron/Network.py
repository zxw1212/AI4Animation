import sys
sys.path.append("../../../PyTorch")

import Library.Utility as utility
import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler

import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, rng, layers, activations, dropout, input_norm, output_norm):
        """
        rng: Random number generator
        layers: List of each layer's size(num of neurons)
        activations: List of activation functions for each layer
        dropout: Dropout rate: drope out some neurons in each training iteration to prevent overfitting
        input_norm: Normalization values for input
        output_norm: Normalization values for output
        """
        super(Model, self).__init__()

        self.rng = rng
        self.layers = layers
        self.activations = activations
        self.dropout = dropout
        self.Xnorm = Parameter(torch.from_numpy(input_norm), requires_grad=False)
        self.Ynorm = Parameter(torch.from_numpy(output_norm), requires_grad=False)
        self.W = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()
        for i in range(len(layers)-1):
            self.W.append(self.weights([self.layers[i], self.layers[i+1]]))
            self.b.append(self.bias([1, self.layers[i+1]]))

    def weights(self, shape):
        # Xavier initialization to get the weights
        # np.prod(shape[-2:]) 得到最后两个元素的乘积,通常表示输入和输出的乘积
        alpha_bound = np.sqrt(6.0 / np.prod(shape[-2:]))
        alpha = np.asarray(self.rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape), dtype=np.float32)
        return Parameter(torch.from_numpy(alpha), requires_grad=True)

    def bias(self, shape):
        return Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        # Normalize example:
        # Xnorm = {
        #     'mean': torch.tensor([0.5, 0.5, 0.5]),  # 输入的均值
        #     'std': torch.tensor([0.2, 0.2, 0.2])    # 输入的标准差
        # }
        # x = (x - Xnorm['mean']) / Xnorm['std']
        x = utility.Normalize(x, self.Xnorm)
        y = x
        for i in range(len(self.layers)-1):
            y = F.dropout(y, self.dropout, training=self.training)
            y = y.matmul(self.W[i]) + self.b[i]
            if self.activations[i] != None:
                y = self.activations[i](y)
        # Renormalize example:
        # Ynorm = {
        # 'mean': torch.tensor([0.5, 0.5, 0.5]),  # 输出的均值
        # 'std': torch.tensor([0.2, 0.2, 0.2])    # 输出的标准差
        # }
        # y = y * Ynorm['std'] + Ynorm['mean']
        return utility.Renormalize(y, self.Ynorm)

if __name__ == '__main__':
    name = "TrackerBody"
    # name = "FutureBody"
    # name = "TrackedUpperBody"
    # name = "UntrackedUpperBody"
    
    directory = "../../Datasets/"+name
    id = name + "_" + utility.GetFileID(__file__)
    load = directory
    save = directory+"/Training_"+id
    utility.MakeDirectory(save)

    InputName = "Input"
    OutputName = "Output"
    InputFile = load + "/" + InputName + ".bin"
    OutputFile = load + "/" + OutputName + ".bin"
    Xshape = utility.LoadTxtAsInt(load + "/" + InputName + "Shape.txt", True)
    Yshape = utility.LoadTxtAsInt(load + "/" + OutputName + "Shape.txt", True)
    Xnorm = utility.LoadTxt(load + "/" + InputName + "Normalization.txt", True)
    Ynorm = utility.LoadTxt(load + "/" + OutputName + "Normalization.txt", True)

    seed = 23456
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    
    epochs = 150
    batch_size = 32
    dropout = 0.25

    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2

    sample_count = Xshape[0]
    input_dim = Xshape[1]
    output_dim = Yshape[1]

    hidden_dim = 512

    layers = [input_dim, hidden_dim, hidden_dim, output_dim]
    activations = [F.elu, F.elu, None]

    print("Network Structure:", layers)

    network = Model(
        rng=rng,
        layers=layers,
        activations=activations,
        dropout=dropout,
        input_norm=Xnorm,
        output_norm=Ynorm
    )
    if torch.cuda.is_available():
        print('GPU found, training on GPU...')
        network = network.cuda()
    else:
        print('No GPU found, training on CPU...')
    
    # weight_decay: 权重衰减, 用于L2正则化惩罚较大的权重, 以防过拟合
    # learning_rate: 控制参数更新步长, 较大的学习率可能导致不稳定的训练, 较小的学习率可能导致训练速度过慢, 通常设置为1e-3
    # epoch_size: 一个epoch的样本数量, 通常是训练集的大小
    # batch_size: 每次迭代中使用的样本数量, 通常是2的幂次方, 例如32, 64, 128, 较大的batch_size会加速训练但可能导致内存不足
    # restart_period: 重启周期, 学习率调度器在多少个epoch后重启, 用于学习率调度策略
    # restart_mult: 重启倍数, 每次重启后, 重启周期乘以重启倍数, 用于学习率调度策略
    # policy: 学习率调度策略, 通常是cosine(cos退火策略), 也可以是linear, exp, poly, 学习率在每个epoch内按照cosine形式逐渐减小.
    optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=batch_size, epoch_size=sample_count, restart_period=restart_period, t_mult=restart_mult, policy="cosine", verbose=True)
    loss_function = torch.nn.MSELoss()

    # 样本的索引, 0~sample_count-1
    I = np.arange(sample_count)
    for epoch in range(epochs):
        # 开始每个epoch的训练,一共训练epochs次

        # 学习率调度器的epoch_step方法, 用于更新学习率
        scheduler.step()
        # 打乱样本的索引, 保证每个epoch的训练样本顺序不同
        np.random.shuffle(I)
        error = 0.0
        for i in range(0, sample_count, batch_size):
            # 开始每个batch的训练, 一共训练sample_count/batch_size次
            print('Progress', round(100 * i / sample_count, 2), "%", end="\r")
            train_indices = I[i:i+batch_size]

            xBatch = utility.ReadBatchFromFile(InputFile, train_indices, input_dim)
            yBatch = utility.ReadBatchFromFile(OutputFile, train_indices, output_dim)
            
            # 网络前向传播, 得到预测值
            yPred = network(xBatch)

            loss = loss_function(utility.Normalize(yPred, network.Ynorm), utility.Normalize(yBatch, network.Ynorm))
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播,计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 学习率调度器的batch_step方法, 用于更新学习率
            scheduler.batch_step()

            error += loss.item()
        
        # 导出模型为ONNX格式
        utility.SaveONNX(
            path=save + '/' + id + '_' + str(epoch+1) + '.onnx',
            model=network,
            input_size=(torch.zeros(1, input_dim)),
            input_names=['X'],
            output_names=['Y']
        )
        # 打印每个epoch的平均损失
        print('Epoch', epoch+1, error/(sample_count/batch_size))