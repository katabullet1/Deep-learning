
import torch.nn as nn 
import torchvision.models as models
import torch.nn.functional as F
#Building a classifier network
class  VGG_networks(nn.Module):
    
 
    def __init__(self,  in_features, hidden_features, out_features, drop_prob=0.7):
        
        


        super().__init__()
        
        self.h1 = nn.Linear(in_features, hidden_features)
        self.h2 = nn.Linear(hidden_features, hidden_features)
        self.h3 = nn.Linear(hidden_features, hidden_features)
        self.h4 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(p=drop_prob)
    def forward(self, x):
        x = self.drop(F.relu(self.h1(x)))
        x = self.drop(F.relu(self.h2(x)))
        x = self.drop(F.relu(self.h3(x)))
        x = self.h4(x)
        
        x = F.log_softmax(x, dim=1)
        return x    
    
 