import torch
import copy
from torch import nn
import torch.nn.functional as F



'''
RAAN Model 
    [input]  
        input : args.batch_size * args.input_samples * args.input_size
    [output]
        all_outputs : args.batch_size * args.temporal_attention_filters
        all_atts : args.batch_size * args.input_samples * args.temporal_attention_filters * 1
'''
class RAAN(nn.Module):
    def __init__(self, args, uniform=False):
        super(RAAN, self).__init__()

        # Input Feature size
        if args.spatial_attention:
            self.input_feature_size = args.spatial_attention_f_maps
        else:
            self.input_feature_size = args.input_size

        # Uniform
        if uniform:
            self.attention = False
            self.temporal_attention_filters = 1
            self.temporal_attention_samples = args.input_samples
        else:
            self.attention = True
            self.temporal_attention_filters = args.temporal_attention_filters
            self.temporal_attention_samples = args.temporal_attention_samples

        # Spatial Model
        if args.input_feature == '2d':
            self.spatial_pool = True
            self.spatial_attention = args.spatial_attention
        else:
            self.spatial_pool = False
            self.spatial_attention = False

        # Temporal model
        if args.temporal_model:
            self.feature_size = args.num_f_maps
            self.tcn = SingleStageTCN(args, self.input_feature_size)
        else:
            self.feature_size = self.input_feature_size
            self.tcn = False

        # All
        self.input_samples = args.input_samples
        self.temporal_att_output = args.temporal_attention_size
        self._prepare_raan_model()


    def _prepare_raan_model(self):
        
        # Uniform pooling layer
        self.uniform_pooling = nn.AdaptiveAvgPool1d(self.input_samples)

        # Spatial pooling layer
        self.spatial_pooling = nn.AdaptiveAvgPool2d((1,1))

        # Spatial attention layer
        self.spatial_att_net = Spatial_Attention(self.input_feature_size)

        # Temporal pooling layer 
        self.temporal_pooling = nn.AdaptiveAvgPool1d(self.temporal_attention_samples)

        # Temporal attention layer
        self.temporal_att_net = nn.ModuleList()
        for i in range(0, self.temporal_attention_filters):
            # softmax for temporal dimention
            self.temporal_att_net.append(nn.Sequential(
                nn.Linear(self.feature_size, self.temporal_att_output),
                nn.ReLU(),
                nn.Linear(self.temporal_att_output, 1),
                nn.Softmax(dim=1)))

        # FC layer
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 1),
            nn.Tanh())


    def forward(self, input):

        # Temporal Pooling
        if self.spatial_pool:
            input = self.uniform_pooling(input.view(input.size(0), 
                                         input.size(1), -1).permute(0,2,1)).permute(0,2,1).view(input.size(0),
                                                                                                -1, input.size(2), input.size(3), input.size(4))
            # Spatial Attention
            if self.spatial_attention:
                spatial_attention, input = self.spatial_att_net(input)
            else:
                spatial_attention = None
            # Spatial Pooling
            input = self.spatial_pooling(input.reshape(-1, input.size(2), input.size(3), input.size(4))).view(input.size(0), input.size(1), -1)
        else:
            input = self.uniform_pooling(input.permute(0,2,1)).permute(0,2,1)
            spatial_attention = None
            
        # Temporal Model
        if self.tcn:
            input = self.tcn(input)

        # Temporal Pooling (args.input_samples → args.temporal_attention_samples)
        input = self.temporal_pooling(input.permute(0,2,1)).permute(0,2,1)

        # Temporal Attention
        if self.attention:
            att_list = []
            # input into each attention filter
            for i in range(0, self.temporal_attention_filters):
                att_list.append(self.temporal_att_net[i](input))
            all_atts = torch.stack(att_list, 2)
        else:
            # even attention weight for each samples
            all_atts = torch.ones((input.size(0),self.input_samples, self.temporal_attention_filters, 1)).cuda() * (1.0/self.input_samples)

        # Feature * Attention weight  -→  Fully Conv
        outputs = []
        for i in range(0, self.temporal_attention_filters):
            # attention weighted : args.batch_size * args.input_samples * self.feature_size
            tmp_outputs = torch.mul(input, all_atts[:,:,i,:])
            # sum all samples : args.batch_size * self.feature_size
            tmp_outputs = tmp_outputs.sum(1)
            outputs.append(self.fc(tmp_outputs).view(-1)*2)
        all_outputs = torch.stack(outputs, 1)

        return all_outputs, all_atts, spatial_attention



'''
Temporal Convolution Network
    [input]  
        x : args.batch_size * args.input_samples * args.input_size
    [output]
        output : args.batch_size * args.input_samples * args.num_f_maps
'''
class SingleStageTCN(nn.Module):
    def __init__(self, args, input_feature):
        super(SingleStageTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_feature, args.num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, args.num_f_maps, args.num_f_maps)) for i in range(args.num_layers)])


    #def forward(self, x, mask):
    def forward(self, x):
        out = self.conv_1x1(x.permute(0,2,1))
        for layer in self.layers:
            #out = layer(out, mask)
            out = layer(out)
        #output = (out * mask).permute(0,2,1)
        output = out.permute(0,2,1)
        return output



class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()


    #def forward(self, x, mask):
    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        #return (x + out) * mask
        return (x + out) 



'''
Attention Layer (Relu)
    [input]  
        x : args.batch_size * args.input_samples * args.input_size * h_size(default:7)* w_size(default:7)
    [output]
        attention : args.batch_size * args.input_samples * 1 * h_size(default:7)* w_size(default:7)
        output : args.batch_size * args.input_samples * f_maps * h_size(default:7)* w_size(default:7)
'''
class Spatial_Attention(nn.Module):
    def __init__(self, f_maps):
        super(Spatial_Attention, self).__init__()
        
        self.conv1 = nn.Conv2d(512,f_maps,3,1,1)
        self.attention_layer = nn.Sequential(
                               nn.Conv2d(512,f_maps,3,1,1),
                               nn.InstanceNorm2d(f_maps),
                               nn.ReLU()
                               )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        y = x.view(-1, x.size(2), x.size(3), x.size(4))
                       
        z = self.attention_layer(y)
        
        attention = self.softmax(torch.mean(y, dim=1).view(y.size(0), -1)).view(x.size(0), x.size(1), 1, -1, y.size(-1))

        output = F.relu((self.conv1(y).view(x.size(0), x.size(1), -1, x.size(3), x.size(4)))*attention)

        return attention, output



if __name__ == '__main__':
    import yaml, os
    from addict import Dict
    args = Dict(yaml.safe_load(open(os.path.join('args', 'tcn.yaml'))))

    input_ten = torch.randn((16, 400, 1024))
    print(input_ten.size())

    model = RAAN(args, uniform = False)
    outputs, attention = model(input_ten)

    print(outputs.size())
    print(attention.size())