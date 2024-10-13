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
from cbir.utils.grid import grid

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

# BEGIN EVALUATION
eval = pd.DataFrame(
    columns=[
        "model",
        "metric",
        "map@1",
        "map@5",
        "map@10",
        "hit_rate@1",
        "hit_rate@5",
        "hit_rate@10",
        "avg_indexing_time",
        "avg_retrieval_time",
    ]
)

models = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
knn_metrics = ["euclidean", "cosine"]
for model, metric in grid(models, knn_metrics):
    print("Evaluate for model: ", model, " with knn metric: ", metric)

    # Initialization
    resnet = ResNetExtractor(model=model, device="cuda")
    array_store = NPArrayStore(retrieve=KNNRetrieval(metric=metric))
    cbir = CBIR(resnet, array_store)

    # Indexing
    start = time()
    for images, labels in tqdm(dataloader, desc="Indexing"):
        images = images.numpy()
        cbir.indexing(images)
    avg_indexing_time = round((time() - start) / len(dataset), 6)

    # Retrieval
    start = time()
    rs = []
    ground_truth = []
    for images, labels in tqdm(testloader, desc="Retrieval"):
        images = images.numpy()
        rs.extend(cbir.retrieve(images, k=10))
        ground_truth.extend(labels)
    avg_retrieval_time = round((time() - start) / len(dataset), 6)

    # Evaluation
    ap1 = []
    hit1 = []
    ap5 = []
    hit5 = []
    ap10 = []
    hit10 = []
    for r, g in zip(rs, ground_truth):
        predicted = []
        for i in r:
            predicted.append(i.index)
        class_preds = np.take(dataset.targets, predicted, axis=0)
        ap1.append(average_precision(class_preds.tolist(), [g.tolist()], 1))
        hit1.append(hit_rate(class_preds.tolist(), [g.tolist()], 1))
        ap5.append(average_precision(class_preds.tolist(), [g.tolist()], 5))
        hit5.append(hit_rate(class_preds.tolist(), [g.tolist()], 5))
        ap10.append(average_precision(class_preds.tolist(), [g.tolist()], 10))
        hit10.append(hit_rate(class_preds.tolist(), [g.tolist()], 10))

    map1 = round(np.mean(ap1), 6)
    avg_hit1 = round(np.mean(hit1), 6)
    map5 = round(np.mean(ap5), 6)
    avg_hit5 = round(np.mean(hit5), 6)
    map10 = round(np.mean(ap10), 6)
    avg_hit10 = round(np.mean(hit10), 6)

    new_row = pd.DataFrame(
        {
            "model": [model],
            "metric": [metric],
            "map@1": [map1],
            "map@5": [map5],
            "map@10": [map10],
            "hit_rate@1": [avg_hit1],
            "hit_rate@5": [avg_hit5],
            "hit_rate@10": [avg_hit10],
            "avg_indexing_time": [avg_indexing_time],
            "avg_retrieval_time": [avg_retrieval_time],
        }
    )
    eval = pd.concat([eval, new_row], ignore_index=True)
    print(
        "map@1: ", map1,
        "map@5: ", map5,
        "map@10: ", map10,
        "hit_rate@1: ", avg_hit1,
        "hit_rate@5: ", avg_hit5,
        "hit_rate@10: ", avg_hit10,
        "avg_indexing_time: ", avg_indexing_time,
        "avg_retrieval_time: ", avg_retrieval_time,
    )
eval.to_csv("out/resnet_knn_gpu_eval.csv", index=False)
