import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from core.layers import BaseCNN, BaseRNN, Discriminator
from core.loss import FocalLoss

class snippet_cnnlstm(nn.Module):

    def __init__(self, input_size = 12, 
                 hidden_size = 256, 
                 hidden_output_size = 1, 
                 output_size = 9, 
                 core_model ="CNNLSTM",
                 isCuda = True):

        super(snippet_cnnlstm, self).__init__()
        self.loss_func = FocalLoss()
        self.CELL_TYPE = "LSTM"
        self.INPUT_SIZE = input_size
        self.HIDDEN_SIZE = hidden_size
        self.HIDDEN_OUTPUT_SIZE = hidden_output_size
        self.OUTPUT_SIZE = output_size
        self.CORE = core_model
        self.isCuda = isCuda

        # --- Backbones ---
        
        print(core_model)
        
        if (core_model == "D3CNN"):
            self.BaseCNN = D3CNN(input_size, hidden_size, hidden_output_size, output_size)
        elif(core_model == "HeartNetIEEE"):
            self.BaseCNN = HeartNetIEEE(input_size, hidden_size, hidden_output_size, output_size)
        elif(core_model == "ResCNN"):
            self.BaseCNN = ResCNN(input_size, hidden_size, hidden_output_size, output_size)
        else:
            self.BaseCNN = BaseCNN(input_size, hidden_size, output_size)
            self.BaseRNN = BaseRNN(hidden_size, hidden_size, self.CELL_TYPE).cuda()
        
        self.Discriminator = Discriminator(hidden_size, output_size)
        
        if(isCuda):
            self.BaseCNN = self.BaseCNN.cuda()
            self.Discriminator = self.Discriminator.cuda()
        
        
    def initHidden(self, batch_size, weight_size, isCuda = True):

        """Initialize hidden states"""

        if(isCuda):
            if self.CELL_TYPE == "LSTM":
                h = (torch.zeros(1, batch_size, weight_size).cuda(),
                     torch.zeros(1, batch_size, weight_size).cuda())
            else:
                h = torch.zeros(1, batch_size, weight_size).cuda()
        else:
            if self.CELL_TYPE == "LSTM":
                h = (torch.zeros(1, batch_size, weight_size),
                     torch.zeros(1, batch_size, weight_size))
            else:
                h = torch.zeros(1, batch_size, weight_size)
                
        return h


    def forward(self, X):
        
        hidden = self.initHidden(len(X), self.HIDDEN_SIZE, self.isCuda)
        min_length = 1000
        max_length = 0
        for x in X:
            if min_length > x.shape[0]:
                min_length = x.shape[0]
            if max_length < x.shape[0]:
                max_length = x.shape[0]
        
        tau_list = np.zeros(len(X), dtype=int)
        
        for t in range(max_length):
            slice_input = []
            cnn_input = None # cpu
            for idx, x in enumerate(X):
                slice_input.append(x[tau_list[idx],:,:])
                cnn_input = torch.stack(slice_input, dim=0)

            if(self.CORE == "CNNLSTM"):
                S_t = self.BaseCNN(cnn_input)

                cnn_input.detach()

                S_t = S_t.unsqueeze(0)

                S_t, hidden = self.BaseRNN(S_t, hidden) # Run sequence model

                for idx in range(len(X)):
                    if(tau_list[idx] < X[idx].shape[0]-1):
                        tau_list[idx]+=1
                S_t = hidden[0][-1]
            elif(self.CORE == "CNNLSTM-500"):
                S_t = self.BaseCNN(cnn_input)

                cnn_input.detach()

                S_t = S_t.unsqueeze(0)

                S_t, hidden = self.BaseRNN(S_t, hidden) # Run sequence model

                for idx in range(len(X)):
                    if(tau_list[idx] < X[idx].shape[0]-1):
                        tau_list[idx]+=1
                S_t = hidden[0][-1]
            else:
                S_t = self.BaseCNN(cnn_input)
                cnn_input.detach()
            
            result = self.Discriminator(S_t)
          
        return result 

    def predict(self, X):

        hidden = self.initHidden(len(X), self.HIDDEN_SIZE)
        min_length = 1000
        max_length = 0
        for x in X:
            if min_length > x.shape[0]:
                min_length = x.shape[0]
            if max_length < x.shape[0]:
                max_length = x.shape[0]
        
        tau_list = np.zeros(X.shape[0], dtype=int)
        
        Hidden_states = []
        for t in range(max_length):
            slice_input = []
            cnn_input = None # cpu
            for idx, x in enumerate(X):
                #print(x.shape)
                slice_input.append(torch.from_numpy(x[t,:,:]).float())
                cnn_input = torch.stack(slice_input, dim=0)
            
            if(self.isCuda):
                cnn_input = cnn_input.cuda()

            if(self.CORE == "CNNLSTM"):
                S_t = self.BaseCNN(cnn_input)
                cnn_input.detach()
                S_t = S_t.unsqueeze(0)

                S_t, hidden = self.BaseRNN(S_t,hidden) # Run sequence model

                S_t = hidden[0][-1]
            elif(self.CORE == "CNNLSTM-500"):
                S_t = self.BaseCNN(cnn_input)
                cnn_input.detach()
                S_t = S_t.unsqueeze(0)

                S_t, hidden = self.BaseRNN(S_t,hidden) # Run sequence model

                S_t = hidden[0][-1]
            else:
                S_t = self.BaseCNN(cnn_input)
                cnn_input.detach()
                S_t = S_t.unsqueeze(0)

                S_t, hidden = self.BaseRNN(S_t,hidden) # Run sequence model

                S_t = hidden[0][-1]
                
            Hidden_states.append(S_t.cpu().detach().numpy())
            
        return Hidden_states
    
    def inference(self, X, hidden):
        
        cnn_input = torch.from_numpy(X).float()
            
        if(self.isCuda):
            cnn_input = cnn_input.cuda()

        if(self.CORE == "CNNLSTM"):
            S_t = self.BaseCNN(cnn_input)
            cnn_input.detach()
            S_t = S_t.unsqueeze(0)
            S_t, hidden = self.BaseRNN(S_t,hidden) # Run sequence model
            S_t = hidden[0][-1]
        elif(self.CORE == "CNNLSTM-500"):
            S_t = self.BaseCNN(cnn_input)
            cnn_input.detach()
            S_t = S_t.unsqueeze(0)
            S_t, hidden = self.BaseRNN(S_t,hidden) # Run sequence model
            S_t = hidden[0][-1]
        else:
            S_t = self.BaseCNN(cnn_input)
            cnn_input.detach()
            S_t = S_t.unsqueeze(0)
            S_t, hidden = self.BaseRNN(S_t,hidden) # Run sequence model
            S_t = hidden[0][-1]
                
        return S_t.cpu().detach().numpy(), hidden




