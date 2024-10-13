import warnings
from time import time

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from cbir import *
from cbir.pipeline import *

# Ignore all warnings
warnings.filterwarnings("ignore")

# Load data
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        # transforms.Normalize(mean, std),
        # lambda x: torch.flip(x, [1]),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
dataset = torchvision.datasets.ImageFolder(
    root="./data/caltech101/train",
    transform=transform,
)

valset = torchvision.datasets.ImageFolder(
    root="./data/caltech101/val",
    transform=transform,
)

testset = torchvision.datasets.ImageFolder(
    root="./data/caltech101/test",
    transform=transform,
)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=128, shuffle=False, num_workers=2
)

valloader = torch.utils.data.DataLoader(
    valset, batch_size=128, shuffle=False, num_workers=2
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2
)

eval = pd.DataFrame(
    columns=["options", "map", "hit_rate", "avg_indexing_time", "avg_retrieval_time"]
)

# BEGIN EVALUATION
for options in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
    print("Evaluate for model: ", options)

    # Initialization
    resnet = ResNetExtractor(model=options, device="cuda")
    array_store = NPArrayStore(retrieve=KNNRetrieval(metric="cosine"))
    cbir = CBIR(resnet, array_store)

    # Indexing
    start = time()
    for images, labels in tqdm(dataloader, desc="Indexing"):
        # images = (images.numpy().transpose(0,2,3,1) * 255).astype(np.uint8)
        images = images.numpy()
        cbir.indexing(images)
    avg_indexing_time = round((time() - start) / len(dataset), 6)

    # Retrieval
    start = time()
    rs = []
    ground_truth = []
    for images, labels in tqdm(testloader, desc="Retrieval"):
        # images = (images.numpy().transpose(0,2,3,1) * 255).astype(np.uint8)
        images = images.numpy()
        rs.extend(cbir.retrieve(images, k=10))
        ground_truth.extend(labels)
    avg_retrieval_time = round((time() - start) / len(dataset), 6)

    # Evaluation
    ap = []
    hit = []
    for r, g in zip(rs, ground_truth):
        predicted = []
        for i in r:
            predicted.append(i.index)
        class_preds = np.take(dataset.targets, predicted, axis=0)
        ap.append(average_precision(class_preds.tolist(), [g.tolist()], 10))
        hit.append(hit_rate(class_preds.tolist(), [g.tolist()], 10))

    map = round(np.mean(ap), 6)
    avg_hit = round(np.mean(hit), 6)
    
    new_row = pd.DataFrame(
        {
            "options": [options],
            "map": [map],
            "hit_rate": [avg_hit],
            "avg_indexing_time": [avg_indexing_time],
            "avg_retrieval_time": [avg_retrieval_time],
        }
    )
    eval = pd.concat([eval, new_row], ignore_index=True)
    print(
        "map: ", map,
        "hit_rate: ", avg_hit,
        "avg_indexing_time: ", avg_indexing_time,
        "avg_retrieval_time: ", avg_retrieval_time,
    )
eval.to_csv("out/resnet_eval.csv")
