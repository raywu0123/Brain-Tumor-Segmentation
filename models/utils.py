import torch


def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        weights = torch.tensor(weights, requires_grad=False)
        if torch.cuda.is_available:
            weights = weights.cuda()
        loss = weights[1] * (target * torch.log(output + 1e-8)) + \
            weights[0] * ((1 - target) * torch.log(1 - output + 1e-8))
    else:
        loss = target * torch.log(output + 1e-8) + (1 - target) * torch.log(1 - output + 1e-8)

    return torch.neg(torch.mean(loss))
