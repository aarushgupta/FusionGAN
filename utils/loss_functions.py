import torch

def lossIdentity(real_pair, fake_pair):
	"""
		Calculate the Identity Loss for the generator and discriminator outputs 
		as mentioned in the paper.
	"""
    batch_size = real_pair.size()[0]
    real_pair = 1 - real_pair
    real_pair = real_pair ** 2
    fake_pair  = fake_pair ** 2
    real_pair = torch.sum(real_pair)
    fake_pair = torch.sum(fake_pair)
    return (real_pair + fake_pair) / batch_size


def lossShape(x, y):
	"""
		Calculate the L1 loss as mentioned in the paper
	"""
    batch_size = x.size()[0]
    diff = torch.abs(x - y)
    diff = torch.sum(diff) / batch_size
    return diff