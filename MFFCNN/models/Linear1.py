import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch.fft as fft



from layers.Linear_backbone_fft_multi import Model as model1








class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()



        self.is_decomposition = 0
        self.model_1=model1(configs)
        self.window=7
        self.n_layers_1=1
        self.seq_len = configs.seq_len
        self.pre_len=configs.pred_len
        self.enc_in=configs.enc_in
        self.layers_1=nn.ModuleList([ self.model_1  for i in range(self.n_layers_1)])
        self.d_model = configs.d_model
        self.relu = nn.GELU()
        self.drop = nn.Dropout(0.1)
        



        self.softmax = nn.Softmax(dim=-1)
       

        self.revin=True
        self.affine=configs.affine
        self.subtract_last=configs.subtract_last
        self.alpha = nn.Parameter(torch.tensor([0.95])) 
        if self.revin: 
            self.revin_layer = RevIN(self.enc_in, affine=self.affine, subtract_last=self.subtract_last)
        self.norm='layer'



        
        
        self.i_linear = nn.Linear(self.d_model, self.pre_len)
        self.i_linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pre_len,self.seq_len]))

    
    def forward(self, x):
      device=x.device
      org_data=x
      if self.revin: #普通标准化  patch+revin mse:0.3980427086353302, mae:0.41247278451919556,  rse:0.5982466340065002
            x = self.revin_layer(x, 'norm')#x: [Batch, Input length, Channel]

      # y = nn.Linear(96, 336).to('cuda')
      # x = x.permute(0,2,1)
      # x = y(x)
      # x = x.permute(0,2,1)

      

        
               

    
        
      org_data=x

        
        
        

    
    
    


      
        
    
       
        

        #print(x.shape)
      if self.is_decomposition: #mse:0.39785265922546387, mae:0.4144216775894165, rse:0.5981037616729736
        pass
      else: #否则不用序列拆分，那么直接先交换数据维度，让input数据层放最后，然后将数据放入模型，最后在把原来数据的位#置放回来
            #print('-------------------无时序分解-----------------------------')
            for model in self.layers_1:
                x= model(x,)
             #下面是另一个模型
#             for model in self.layers: 
#                 x= model(x)

#             x=self.softmax(x)
#             x=org_data+x
#             for model in self.layers: 
#                 x= model(x)

            x=x.permute(0,2,1)
            x=self.i_linear(x)



            # x =self.relu(x)
            # org_data = org_data.permute(0,2,1)
            # x = x + org_data
            #
            #
            #
            # x=self.i_linear(x)


#             x=torch.cat((org_data,x),dim=-1)
#             x=self.i_linear1(x)
           # x=x+org_data
           # x=self.relu(x)
        #    x=x+org_data
            x=x.permute(0,2,1)
 #     x=x.permute(0,2,1)
  #    x=self.Linear1(x) #到时候调回2048           #x=x+org_data
 #     x=x.permute(0,2,1)
         #x=x+org_data
      if self.revin:  #revin=True表明对数据进行了标准化,数据已经经过组干模型处理了，需要对其反标准
          x = self.revin_layer(x, 'denorm')
           # x: [Batch, Input length, Channel]

      return x  #mse:0.39124175906181335, mae:0.40738728642463684, rse:0.5931137800216675
