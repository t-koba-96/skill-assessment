import torch

def diversity_loss(attention, args, device):
    attention_t = torch.transpose(attention, 1, 2)
    num_features = attention.shape[1]
    res = torch.matmul(attention_t.view(-1, args.num_filters, num_features), attention.view(-1, num_features, args.num_filters)) - torch.eye(args.num_filters).to(device)
    res = res.view(-1, args.num_filters*args.num_filters)
    return torch.norm(res, p=2, dim=1).sum() / attention.size(0)

# First V
def multi_rank_loss(input_a_1, input_a_2, input_b_1, input_b_2, target, margin, device):
    inter1, _ = torch.min((input_a_1 - input_a_2), dim=1)
    inter2 = (input_b_1 - input_b_2)
    inter = -target * (inter1.view(-1) - inter2.view(-1)) + torch.ones(input_a_1.size(0)).to(device)*margin
    losses = torch.max(torch.zeros(input_a_1.size(0)).to(device), inter)
    return losses.sum()/input_a_1.size(0)

# Second V
def multi_rank_loss_v2(input_a_1, input_a_2, input_b_1, input_b_2, target, margin, device):
    inter1, _ = torch.min((input_a_1 - input_a_2), dim=1)
    inter2 = (input_b_1 - input_b_2)
    inter = -target * (inter1.view(-1) - inter2.view(-1)) + torch.ones(input_a_1.size(0)).to(device)*margin
    losses = torch.max(torch.zeros(input_a_1.size(0)).to(device), inter)
    return losses.sum()/input_a_1.size(0)