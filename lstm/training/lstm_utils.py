import numpy as np 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    def __init__(self, np_X_file_paths,np_y_file_paths):
        self.tag2index={'Poision':0,'Normal':1,'queenless':2,'normal':1}
        self.X_files = np.load(np_X_file_paths)
        self.y_files = np.load(np_y_file_paths)
        #self.time_files = np.load(np_time_file_paths)
        #self.tw = train_window
    
    def __getitem__(self, index):
        x = self.X_files[index]
        x = np.resize(x,(40,-1))
        
        x = np.transpose(x)
        '''
        time = self.time_files[index]
        time = np.repeat(time,x.shape[0])
        time = np.reshape(time,(x.shape[0],-1))
        
        x = np.concatenate((x,time),axis=1)
        x = torch.from_numpy(x).float()'''
        x = torch.from_numpy(x).float()
        
        y = self.y_files[index]
        y = self.tag2index[y]
        y = torch.tensor(y)
        y = y.type(torch.float32)
        
        return x,y
    
    def __len__(self):
        #return len(self.X_files)-self.tw
        return len(self.X_files)
    
'''    
    

class MyDataset(Dataset):
    def __init__(self, np_X_file_paths,np_y_file_paths,train_window):
        self.tag2index={'normal':0,'queenless':1}
        self.X_files = np.load(np_X_file_paths)
        self.y_files = np.load(np_y_file_paths)
        self.tw = train_window
    
    def __getitem__(self, index):
        x = self.X_files[index:index+self.tw]
        x = np.resize(x,(40,-1))
        x = np.transpose(x)
        x = torch.from_numpy(x).float()
        
        y = self.y_files[index]
        y = self.tag2index[y]
        y = torch.tensor(y)
        y = y.type(torch.float32)
        
        return x,y
    
    def __len__(self):
        return len(self.X_files)-self.tw

    
class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
'''
    
class LSTM(nn.Module):
    def __init__(self, input_size=41, hidden_layer_size=10, output_size=1,bidirectional=True):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        
        self.drop_en = nn.Dropout(p=0.6)

        self.lstm = nn.LSTM(input_size,hidden_layer_size,bidirectional=bidirectional,batch_first=True)
        
        self.bn2 = nn.BatchNorm1d(hidden_layer_size*2)
        
        self.fc = nn.Linear(hidden_layer_size*2, output_size)
        self.sigmoid = nn.Sigmoid()

        

        #self.hidden_cell = (torch.zeros(1,batch_size,self.hidden_layer_size),
        #                    torch.zeros(1,batch_size,self.hidden_layer_size))

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        
        h = self.drop_en(input_seq)
        
        pack_output,ht = self.lstm(h,None)
        
        last_tensor = pack_output[:,-1,:]
        
        fc_input = self.bn2(last_tensor)
        
        out = self.fc(fc_input)
        
        return self.sigmoid(out)
        