import torch

def diversity_loss(attention, args, device):
    attention_t = torch.transpose(attention, 1, 2)
    num_features = attention.shape[1]
    res = torch.matmul(attention_t.view(-1, args.temporal_attention_filters, num_features), attention.view(-1, num_features, args.temporal_attention_filters)) - torch.eye(args.temporal_attention_filters).to(device)
    res = res.view(-1, args.temporal_attention_filters*args.temporal_attention_filters)
    return torch.norm(res, p=2, dim=1).sum() / attention.size(0)


def disparity_loss(input_a_1, input_a_2, input_b_1, input_b_2, target, margin, device, version):
    inter1, _ = torch.min((input_a_1 - input_a_2), dim=1)
    inter2 = (input_b_1 - input_b_2)
    if version == "v1":
        inter = -target * (inter1.view(-1) - inter2.view(-1)) + torch.ones(input_a_1.size(0)).to(device)*margin
    elif version == "v2":
        inter = -target * (inter1.view(-1) / inter2.view(-1)) + torch.ones(input_a_1.size(0)).to(device)*margin
    losses = torch.max(torch.zeros(input_a_1.size(0)).to(device), inter)
    return losses.sum()/input_a_1.size(0)
