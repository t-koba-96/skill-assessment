import torch
import copy
from torch import nn
import torch.nn.functional


'''
RAAN Model 
    [input]  
        input : args.batch_size * args.input_samples * args.input_size
    [output]
        all_outputs : args.batch_size * args.attention_filters
        all_atts : args.batch_size * args.input_samples * args.attention_filters * 1
'''
class RAAN(nn.Module):
    def __init__(self, args, uniform=False):
        super(RAAN, self).__init__()
        # uniform or not
        if uniform:
            self.attention = False
            self.attention_filters = 1
            self.attention_samples = args.input_samples
        else:
            self.attention = True
            self.attention_filters = args.attention_filters
            self.attention_samples = args.attention_samples

        # Temporal model 
        if args.temporal_model:
            self.feature_size = args.num_f_maps
            self.tcn = SingleStageTCN(args, args.input_size)
        else:
            self.feature_size = args.input_size
            self.tcn = False

        self.input_samples = args.input_samples
        self.att_output = args.attention_size
        self._prepare_raan_model()


    def _prepare_raan_model(self):
        # Temporal pooling layer 
        self.temporal_pooling = nn.AdaptiveAvgPool1d(self.attention_samples)

        # Attention layer
        self.att_net = nn.ModuleList()
        for i in range(0, self.attention_filters):
            # softmax for temporal dimention
            self.att_net.append(nn.Sequential(
                nn.Linear(self.feature_size, self.att_output),
                nn.ReLU(),
                nn.Linear(self.att_output, 1),
                nn.Softmax(dim=1)))

        # FC layer
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 1),
            nn.Tanh())


    def forward(self, input):
        # Temporal Block
        if self.tcn:
            input = self.tcn(input)

        # Temporal Pooling (args.input_samples → args.attention_samples)
        input = self.temporal_pooling(input.permute(0,2,1)).permute(0,2,1)

        # Attention Weight Layer
        if self.attention:
            att_list = []
            # input into each attention filter
            for i in range(0, self.attention_filters):
                att_list.append(self.att_net[i](input))
            all_atts = torch.stack(att_list, 2)
        else:
            # even attention weight for each samples
            all_atts = torch.ones((input.size(0),self.input_samples, self.attention_filters, 1)).cuda() * (1.0/self.input_samples)

        # Feature * Attention weight  -→  Fully Conv
        outputs = []
        for i in range(0, self.attention_filters):
            # attention weighted : args.batch_size * args.input_samples * self.feature_size
            tmp_outputs = torch.mul(input, all_atts[:,:,i,:])
            # sum all samples : args.batch_size * self.feature_size
            tmp_outputs = tmp_outputs.sum(1)
            outputs.append(self.fc(tmp_outputs).view(-1)*2)
        all_outputs = torch.stack(outputs, 1)

        return all_outputs, all_atts



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
        out = torch.nn.functional.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        #return (x + out) * mask
        return (x + out) 



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
