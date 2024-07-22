# code from https://github.com/ts-kim/RevIN, with minor modifications

import torch
import torch.nn as nn
#RevIN的实现包含以下几个方法和属性：

#__init__方法：初始化RevIN层，包括输入特征的数量、数值稳定性参数eps、仿射变换参数affine和是否从最后一个时间步减去的标志subtract_last。如果affine为True，则需要初始化可学习的仿射变换参数。

#forward方法：将输入数据进行归一化或反归一化处理，并返回处理后的结果。mode参数指定了处理模式，可以是'norm'（归一化）或'denorm'（反归一化）。

#_init_params方法：初始化可学习的仿射变换参数，包括权重和偏移。

#_get_statistics方法：计算输入数据的均值和标准差，并保存在mean和stdev属性中。dim2reduce参数指定了需要对哪些维度进行计算，通常是除了Batch维度之外的所有维度。

#_normalize方法：将输入数据进行归一化处理，并返回处理后的结果。如果subtract_last为True，则从最后一个时间步减去一个标量，否则减去均值。然后除以标准差，并进行可学习的仿射变换操作。

#_denormalize方法：将输入数据进行反归一化处理，并返回处理后的结果。如果affine为True，则进行可学习的仿射变换操作。然后乘以标准差，并加上均值。如果subtract_last为True，则加上从最后一个时间步减去的标量。

#使用self.subtract_last的方法是因为在某些应用场景下，序列的最后一个值可能更能代表整个序列的特征，例如在时间序列预测中，最后一个时间步的值通常更能代表未来的趋势。

#使用序列的最后一个值作为均值进行归一化，可以使得每个样本的均值都变为0，从而使得不同样本之间的分布更加一致，更容易训练。此外，使用序列的最后一个值作为均值还可以避免在训练过程中泄露未来信息的问题，因为在训练过程中，模型只能使用之前的数据进行预测，不能使用未来的数据。

#当然，使用普通的均值和标准差也是一种常用的归一化方法，适用于大多数场景。具体使用哪种方法，需要根据具体的应用场景和数据特点进行选择



class RevIN(nn.Module):
    #__init__方法：初始化RevIN层，包括输入特征的数量、数值稳定性参数eps、仿射变换参数affine和是否从最后一个时间步减去的标志subtract_last。如果affine为True，则需要初始化可学习的仿射变换参数。
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features  #维度数，需要给每个维度创造一个w,b
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x
#_init_params方法：初始化可学习的仿射变换参数，包括权重和偏移。
    def _init_params(self):
        # initialize RevIN params: (C,) #初始化RevIN参数
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
#计算输入数据的均值和标准差，并保存在mean和stdev属性中。dim2reduce参数指定了需要对哪些维度进行计算，通常是除了Batch维度之外的所有维度。
#x[:,-1,:]表示取x张量中每个样本的最后一个时间步的所有变量的值，unsqueeze(1)表示在第二个维度上添加一个维度，将维度从[batch_size, nvars]变为[batch_size, 1, nvars]。
#torch.mean函数是计算张量沿着指定维度的平均值，dim2reduce变量表示除了Batch维度之外的所有维度，keepdim=True表示保持输出的维度和输入的维度一致，detach()表示将结果从计算图中分离出来，不会对梯度造成影响。
#torch.var函数是计算张量沿着指定维度的方差，unbiased=False表示使用有偏方差计算方法，eps表示加上一个极小值，避免出现除零错误。
    def _get_statistics(self, x):  #x=[bs,seq_len,nvars]
        dim2reduce = tuple(range(1, x.ndim-1)) #除了Batch维度之外的所有维度
        if self.subtract_last:#x[:,-1,:]-->[batch,nvars]  取seq_len里面的最后一个数
            self.last = x[:,-1,:].unsqueeze(1)#[batch_size, 1, nvars]
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
#将输入数据进行归一化处理
#这样做可以将输入张量的均值变为0，标准差变为1，从而使得不同样本之间的分布更加一致，更容易训练。
#如果self.affine为True，表示使用仿射变换，那么就将归一化后的张量x乘以self.affine_weight，然后加上self.affine_bias，这样可以进一步改变张量的分布，从而更好地适应模型的需求。
    def _normalize(self, x):
        if self.subtract_last: #将序列的最后一个值当成mean
            x = x - self.last 
        else: #普通方法
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x
#将输入数据进行反归一化处理
    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)#self.eps*self.eps防止结果变为无穷而报错
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
