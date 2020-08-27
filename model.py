import torch
from torch import nn

class RAAN(nn.Module):
    def __init__(self, num_samples, attention, num_filters=3, input_size=1024, fc_output=256):
        super(RAAN, self).__init__()
        self.num_samples = num_samples
        self.attention = attention
        self.num_filters = num_filters
        self.input_size = input_size
        self.fc_output = fc_output

        self._prepare_raan_model()

    def _prepare_raan_model(self):
        self.att_net = nn.ModuleList()

        for i in range(0, self.num_filters):
            # softmax for temporal dimention
            self.att_net.append(nn.Sequential(
                nn.Linear(self.input_size, self.fc_output),
                nn.ReLU(),
                nn.Linear(self.fc_output, 1),
                nn.Softmax(dim=1)))

        self.fc = nn.Sequential(
            nn.Linear(self.input_size, 1),
            nn.Tanh())

    def forward(self, input):
        input = input.view(-1, self.num_samples, self.input_size)
        if self.attention:
            att_list = []
            # input into each attention filter
            for i in range(0, self.num_filters):
                att_list.append(self.att_net[i](input))
            all_atts = torch.stack(att_list, 2)
        else:
            # even attention weight for each samples
            all_atts = torch.ones((input.size(0),self.num_samples, self.num_filters, 1)).cuda() * (1.0/self.num_samples)
        #att = torch.mean(all_atts, 2)
        outputs = []
        for i in range(0, self.num_filters):
            # tmp_outputs : batch * samples * feature (attention weighted)
            tmp_outputs = torch.mul(input, all_atts[:,:,i,:])
            # sum all samples (batch * feature) 
            tmp_outputs = tmp_outputs.sum(1)
            outputs.append(self.fc(tmp_outputs).view(-1)*2)
        all_outputs = torch.stack(outputs, 1)

        # all_outputs : batch * filter
        # all_atts : batch * samples * filter * 1
        return all_outputs, all_atts
