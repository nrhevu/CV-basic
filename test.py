from cbir import *
from cbir.pipeline import *

rgb_histogram = RGBHistogram(n_bin=8, h_type="region")
array_store = NPArrayStore(retrieve=KNNRetrieval(metric="cosine"))

cbir = CBIR(rgb_histogram, array_store)

import torch
import torchvision
import torchvision.transforms as transforms

trainset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=False, transform=transforms.ToTensor()
)
# testset = torchvision.datasets.CIFAR100(
#     root="./data", train=False, download=False, transform=transforms.ToTensor()
# )

# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=4, shuffle=True, num_workers=2
# )
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=4, shuffle=False, num_workers=2
# )

cbir.indexing(trainset.data)