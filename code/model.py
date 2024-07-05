import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def SNN_Block(dim1, dim2, dropout=0.25):
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            # nn.AlphaDropout(p=dropout, inplace=False)
            )


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

class SNN_token(nn.Module):
    def __init__(self, input_dim: int, model_size_omic: str='big', n_classes: int=4, cls_flg: bool=False):
        super(SNN_token, self).__init__()
        self.n_classes = n_classes
        self.cls = cls_flg
        self.size_dict_omic = {'small': [256, 256, 256, 192], 
                               'medium': [512, 512, 512, 512],
                               'big': [1024, 1024, 1024, 1024]}
        
        # Construct Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1]))
        self.fc_omic = nn.Sequential(*fc_omic)
        if self.cls:
            self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)

    def forward(self, x):
        features = self.fc_omic(x)
        if self.cls:
            logits = self.classifier(features)
            Y_hat = torch.argmax(logits, dim=1)  # Predicted class indices
            probs = nn.functional.softmax(logits, dim=1)  # Class probabilities
            return logits, probs, Y_hat
        else:
            return features

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.fc_omic = nn.DataParallel(self.fc_omic, device_ids=device_ids).to('cuda')
            if self.cls:
                self.classifier = nn.DataParallel(self.classifier, device_ids=device_ids).to('cuda')
            
        else:
            self.fc_omic = self.fc_omic.to(device)
            if self.cls:
                self.classifier = self.classifier.to(device)

          
class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):
        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x

class Attention_Gated(nn.Module):
    def __init__(self, L=192, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class ABMIL_VPT(nn.Module):
    def __init__(self, n_classes: int=4, L:int=192, prompt_length=4):
        super(ABMIL_VPT, self).__init__()
        self.prompt = nn.Parameter(torch.randn(prompt_length, 192))
        self.fc_1 = nn.Linear(192, L)
        self.n_classes = n_classes
        self.attention_net = Attention_Gated(L, D = 256)
        self.classifiers = Classifier_1fc(L, n_classes, droprate=0.25)
        self.predictor = nn.Sequential(nn.Linear(L, L, bias=False),
                                        nn.Dropout(p=0.25),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(L, L)) # output layer
        self.layernorm = nn.LayerNorm(L) 
        
    def forward(self, x): ## x: N x L
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # add prompt
        x = torch.cat([self.prompt, x])
        x = self.fc_1(x)
        AA = self.attention_net(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        embeddings = self.predictor(afeat) 
        embeddings = self.layernorm(embeddings) 
        logits = self.classifiers(afeat) ## K x num_cls
        
        Y_hat = torch.argmax(logits, dim=1)  # Predicted class indices
        probs = nn.functional.softmax(logits, dim=1)  # Class probabilities
        return embeddings, logits, probs, Y_hat


class Discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    def __init__(self, input_size=1024, num_classes=4):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, h):
        y = self.layer(h)
        return y


class cls_wsi(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(cls_wsi, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        logits = self.fc(x) ## K x num_cls
        
        Y_hat = torch.argmax(logits, dim=1)  # Predicted class indices
        probs = nn.functional.softmax(logits, dim=1)  # Class probabilities
        return logits, probs, Y_hat
    

