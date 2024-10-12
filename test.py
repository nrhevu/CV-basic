import torch
import torchvision
import torchvision.transforms as transforms

trainset = torchvision.datasets.ImageFolder(
    root="./data/caltech-101",
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((256, 256))]
    ),
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)

from cbir import *
from cbir.pipeline import *

rgb_histogram = RGBHistogram(n_bin=8, h_type="region")
resnet = ResNetExtractor(model = "resnet152", pick_layer="avg")
array_store = NPArrayStore(retrieve=KNNRetrieval(metric="cosine"))

cbir = CBIR(resnet, array_store)

for images, labels in tqdm(trainloader):
    cbir.indexing(images.permute(0, 1, 2, 3).numpy())