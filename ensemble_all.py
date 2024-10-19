import gc
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
from cbir.utils.ensemble import ensemble_search
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

eval = pd.DataFrame(
    columns=[
        "k",
        "distance2score",
        "weight(sift vs color)",
        "map@1",
        "map@5",
        "map@10",
        "hit_rate@1",
        "hit_rate@5",
        "hit_rate@10",
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@100",
        "recall@1000",
        "avg_indexing_time",
        "avg_retrieval_time",
    ]
)

# BEGIN EVALUATION
siftbow = SIFTBOWExtractor(mode="tfidf")
sift_array_store = NPArrayStore(retrieve=KNNRetrieval(metric="manhattan"))

rgb_histogram = RGBHistogram(n_bin=4, h_type="region")
color_array_store = NPArrayStore(retrieve=KNNRetrieval(metric="cosine"))

resnet = ResNetExtractor(model="resnet152", device="cuda")
resnet_array_store = NPArrayStore(retrieve=KNNRetrieval(metric="cosine"))
# Fitting siftbow with train data
print("Fitting BOW for n_clusters kmeans: ", 96)
train_img = []
for images, labels in tqdm(valloader):
    images = (images.numpy().transpose(0,2,3,1) * 255).astype(np.uint8)
    train_img.append(images)

train_img = np.concatenate(train_img)
siftbow.fit(train_img, k=96)

cbir_sift = CBIR(siftbow, sift_array_store)
cbir_color = CBIR(rgb_histogram, color_array_store)
cbir_resnet = CBIR(resnet, resnet_array_store)

# Indexing
start = time()
for images, labels in tqdm(dataloader, desc="Indexing"):
    cbir_resnet.indexing(images.numpy())
    images = (images.numpy().transpose(0,2,3,1) * 255).astype(np.uint8)
    cbir_sift.indexing(images)
    cbir_color.indexing(images)
avg_indexing_time = round((time() - start) / len(dataset), 6)

ks = [len(dataset)] # Top K each algorithm
d2ss = ["exp"] # Distance to Score Function
weights = [
    (round(0.8 * 0.2, 2), round(0.2 * 0.2, 2), 0.8),
    (round(0.8 * 0.4, 2), round(0.2 * 0.4, 2), 0.6),
    (round(0.8 * 0.5, 2), round(0.2 * 0.5, 2), 0.5),
    (round(0.8 * 0.6, 2), round(0.2 * 0.6, 2), 0.4),
    (round(0.8 * 0.8, 2), round(0.2 * 0.8, 2), 0.2),
]  # Weights for each algorithm
cache = {"sift":{},
         "color": {},
         "resnet": {}}
for k, d2s, weight in grid(ks, d2ss, weights):
    print("Evaluate for init k: ", k, " with d2s: ", d2s, " with weight: ", weight, " (sift vs color vs resnet)")
    # Retrieval
    start = time()
    rs = []
    ground_truth = []
    image_count = 0
    for images, labels in tqdm(testloader, desc="Retrieval"):
        images = (images.numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        for image in images:
            try:
                cbir_sift_result = cache["sift"][f"{image_count}-{k}-{d2s}"]
                cbir_color_result = cache["color"][f"{image_count}-{k}-{d2s}"]
                cbir_resnet_result = cache["resnet"][f"{image_count}-{k}-{d2s}"]
            except KeyError:
                cbir_sift_result = cbir_sift.retrieve(image, k=k, distance_transform=d2s)
                cache["sift"][f"{image_count}-{k}-{d2s}"] = cbir_sift_result
                cbir_color_result = cbir_color.retrieve(image, k=k, distance_transform=d2s)
                cache["color"][f"{image_count}-{k}-{d2s}"] = cbir_color_result
                cbir_resnet_result = cbir_resnet.retrieve(
                    np.expand_dims(image.transpose(2, 0, 1), 0) / 255,
                    k=k,
                    distance_transform=d2s,
                )[0]
                cache["resnet"][f"{image_count}-{k}-{d2s}"] = cbir_resnet_result
            image_count += 1
            rs.append(
                ensemble_search(
                    cbir_sift_result,
                    cbir_color_result,
                    cbir_resnet_result,
                    weights=weight,
                    datalength=len(dataset),
                    k = 1000
                )
            )
        ground_truth.extend(labels)
    avg_retrieval_time = round((time() - start) / len(dataset), 6)

    # Evaluation
    ap1 = []
    hit1 = []
    recall1 = []
    ap5 = []
    hit5 = []
    recall5 = []
    ap10 = []
    hit10 = []
    recall10 = []
    recall100 = []
    recall1000 = [] 
    for r, g in zip(rs, ground_truth):
        predicted = []
        for i in r:
            predicted.append(i.index)
        class_preds = np.take(dataset.targets, predicted, axis=0)
        ap1.append(average_precision(class_preds.tolist(), [g.tolist()], 1))
        hit1.append(hit_rate(class_preds.tolist(), [g.tolist()], 1))
        recall1.append(recall(predicted, np.where(np.isin(np.array(dataset.targets), [g.tolist()]))[0], 1))
        ap5.append(average_precision(class_preds.tolist(), [g.tolist()], 5))
        hit5.append(hit_rate(class_preds.tolist(), [g.tolist()], 5))
        recall5.append(recall(predicted, np.where(np.isin(np.array(dataset.targets), [g.tolist()]))[0], 5))
        ap10.append(average_precision(class_preds.tolist(), [g.tolist()], 10))
        hit10.append(hit_rate(class_preds.tolist(), [g.tolist()], 10))
        recall10.append(recall(predicted, np.where(np.isin(np.array(dataset.targets), [g.tolist()]))[0], 10))
        recall100.append(recall(predicted, np.where(np.isin(np.array(dataset.targets), [g.tolist()]))[0], 100))
        recall1000.append(recall(predicted, np.where(np.isin(np.array(dataset.targets), [g.tolist()]))[0], 1000))


    map1 = round(np.mean(ap1), 6)
    avg_hit1 = round(np.mean(hit1), 6)
    avg_recall1 = round(np.mean(recall1), 6)
    map5 = round(np.mean(ap5), 6)
    avg_hit5 = round(np.mean(hit5), 6)
    avg_recall15 = round(np.mean(recall5), 6)
    map10 = round(np.mean(ap10), 6)
    avg_hit10 = round(np.mean(hit10), 6)
    avg_recall10 = round(np.mean(recall10), 6)
    avg_recall100 = round(np.mean(recall100), 6)
    avg_recall1000 = round(np.mean(recall1000), 6)

    # Store evaluation results
    new_row = pd.DataFrame(
        {
            "k": [k],
            "distance2score": [d2s],
            "weight(sift vs color vs resnet)": [weight],
            "map@1": [map1],
            "map@5": [map5],
            "map@10": [map10],
            "hit_rate@1": [avg_hit1],
            "hit_rate@5": [avg_hit5],
            "hit_rate@10": [avg_hit10],
            "recall@1": [avg_recall1],
            "recall@5": [avg_recall15],
            "recall@10": [avg_recall10],
            "recall@100": [avg_recall100],
            "recall@1000": [avg_recall1000],
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
        "recall@1: ", avg_recall1,
        "recall@5: ", avg_recall15,
        "recall@10: ", avg_recall10,
        "recall@100: ", avg_recall100,
        "recall@1000: ", avg_recall1000,
        "avg_indexing_time: ", avg_indexing_time,
        "avg_retrieval_time: ", avg_retrieval_time,
    )

eval.to_csv("out/ensemble_sift_color_resnet_knn_eval.csv", index=False)
