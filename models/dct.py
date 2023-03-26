import torch
import torch.nn as nn
import math
from thop import profile
import torch.nn.functional as F
from torchvision import models
import math

class DCTheadCls(nn.Module):
    def __init__(self, dct_arch='resnet18'):
        super(DCTheadCls, self).__init__()

        if dct_arch == 'resnet18':
            dct = models.resnet18(pretrained=True)
            layer = 'res_layer4'
        elif dct_arch == 'resnet34':
            dct = models.resnet34(pretrained=True)
            layer = 'res_layer4'
        elif dct_arch == 'mobilenet_v2':
            dct = models.mobilenet_v2(pretrained=True)
            layer = 'mv2_features'
        elif dct_arch == 'shufflenet_v2':
            dct = models.shufflenet_v2_x1_0(pretrained=True)
            layer = 'sv2_stage4'

        layer_dim = {'res_layer4':512, 'res_layer3':256, 'res_layer2':128, 'res_layer1':64,  # resnet18 & resnet34
                    'mv2_features':1280,    # mobilenet v2
                    'sv2_stage2':116, 'sv2_stage3':232, 'sv2_stage4':464,    # shufflenet v2
                    }
        
        layer_name = {'res_layer4':'layer4', 'res_layer3':'layer3', 'res_layer2':'layer2', 'res_layer1':'layer1',  # resnet18 & resnet34
                    'mv2_features':'features',    # mobilenet v2
                    'sv2_stage2':'stage2', 'sv2_stage3':'stage3', 'sv2_stage4':'stage4',    # shufflenet v2
                    }

        self.tiny = models._utils.IntermediateLayerGetter(dct,{layer_name[layer]: 'feat'})
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.init_alpha(layer_dim, layer, min_alpha=0.1, max_alpha=2.0, gap=0.1, fc_num=2)
        self.init_beta(layer_dim, layer, min_beta=-0.1, max_beta=0.1, gap=0.02, fc_num=2)
    
    def init_alpha(self, layer_dim, layer, min_alpha=0.1, max_alpha=2.0, gap=0.1, fc_num=2):
        eps = 1e-8
        self.alpha_range = torch.arange(min_alpha, max_alpha+eps, gap).cuda()  # first choice

        if fc_num == 1:
            self.fc1 = nn.Linear(layer_dim[layer], len(self.alpha_range))
        elif fc_num == 2:
            self.fc1 = nn.Linear(layer_dim[layer], layer_dim[layer]//2)
            self.fc2 = nn.Linear(layer_dim[layer]//2, len(self.alpha_range))
        elif fc_num == 3:
            self.fc1 = nn.Sequential(
                nn.Linear(layer_dim[layer], layer_dim[layer]//2),
                nn.ReLU(),
                nn.Linear(layer_dim[layer]//2, layer_dim[layer]//4),
            )
            self.fc2 = nn.Linear(layer_dim[layer]//4, len(self.alpha_range))#, bias=True)
        self.relu = nn.ReLU()
        
    def init_beta(self, layer_dim, layer, min_beta=-0.1, max_beta=0.1, gap=0.02, fc_num=2):
        # beta 
        eps = 1e-8
        self.beta_range = torch.arange(min_beta, max_beta+eps, gap).cuda()  # first choice
        if fc_num == 2:
            self.fc1_beta = nn.Linear(layer_dim[layer], layer_dim[layer]//2)
            self.fc2_beta = nn.Linear(layer_dim[layer]//2, len(self.beta_range))

    def get_alpha(self, x, fc, topk=-1):
        w = fc(x)
        weight = F.softmax(w, dim=1)
        try:
            alpha = weight * self.alpha_range.view(1, -1)
        except:
            alpha = weight * self.alpha_range.view(1, -1).cpu()
        alpha = alpha.sum(dim=1).view(-1, 1, 1, 1)
        return alpha, weight
    
    def get_beta(self, x, fc):
        w = fc(x)
        weight = F.softmax(w, dim=1)
        try:
            beta = weight * self.beta_range.view(1, -1)
        except:
            beta = weight * self.beta_range.view(1, -1).cpu()
        beta = beta.sum(dim=1).view(-1, 1, 1, 1)
        return beta
    
    def alpha_forward(self, x):
        if hasattr(self, 'fc2'):
            x_alpha = self.relu(self.fc1(x))
            alpha, weight = self.get_alpha(x_alpha, self.fc2)
        else:
            alpha, weight = self.get_alpha(x, self.fc1)
        return alpha, weight
    
    def beta_forward(self, x):
        if hasattr(self, 'fc2_beta'):
            x_beta = self.relu(self.fc1_beta(x))
            beta = self.get_beta(x_beta, self.fc2_beta)
        else:
            beta = self.get_beta(x, self.fc1)
        return beta

    def forward(self, x):
        x = self.tiny(x)
        x = x['feat']

        x_pool = self.pool(x)
        x = x_pool.view(x_pool.shape[:-2])

        alpha, weight = self.alpha_forward(x)
        beta = self.beta_forward(x)
        return {'alpha': alpha, 'beta': beta}


class DCTheadReg(nn.Module):
    def __init__(self, dct_arch='resnet18'):
        super(DCTheadReg, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        if dct_arch == 'resnet18':
            dct = models.resnet18(pretrained=True)
            layer = 'res_layer4'
        elif dct_arch == 'resnet34':
            dct = models.resnet34(pretrained=True)
            layer = 'res_layer4'
        elif dct_arch == 'mobilenet_v2':
            dct = models.mobilenet_v2(pretrained=True)
            layer = 'mv2_features'
        elif dct_arch == 'shufflenet_v2':
            dct = models.shufflenet_v2_x1_0(pretrained=True)
            layer = 'sv2_stage4'

        layer_dim = {'res_layer4':512, 'res_layer3':256, 'res_layer2':128, 'res_layer1':64,  # resnet18 & resnet34
                    'mv2_features':1280,    # mobilenet v2
                    'sv2_stage2':116, 'sv2_stage3':232, 'sv2_stage4':464,    # shufflenet v2
                    }
        
        layer_name = {'res_layer4':'layer4', 'res_layer3':'layer3', 'res_layer2':'layer2', 'res_layer1':'layer1',  # resnet18 & resnet34
                    'mv2_features':'features',    # mobilenet v2
                    'sv2_stage2':'stage2', 'sv2_stage3':'stage3', 'sv2_stage4':'stage4',    # shufflenet v2
                    }

        self.tiny = models._utils.IntermediateLayerGetter(dct,{layer_name[layer]: 'feat'})
        self.fc = nn.Linear(layer_dim[layer], 2, bias=True)

        self.alpha_range = 2.0
        self.beta_range = 0.1
        
    def forward(self, x):
        x = self.tiny(x)
        x = x['feat']

        x_pool = self.pool(x)
        x = x_pool.view(x_pool.shape[:-2])

        x = self.fc(x)
        batch = x.shape[0]
        alpha = torch.sigmoid(x[:, :1, ...]) * self.alpha_range
        alpha = alpha.view(-1,1,1,1)

        beta = torch.atan(x[:, 1:, ...]) * self.beta_range / math.pi
        beta = beta.view(-1,1,1,1)
        return {'alpha': alpha, 'beta': beta}


if __name__ == '__main__':
    model = DCTheadReg()
    input = torch.randn(1, 3, 1024, 1024)
    macs, params = profile(model, inputs=(input, ))

    print(f'params: {params/1e6} M')
