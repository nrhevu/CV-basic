from cbir import *
from cbir.pipeline import *

rgb_histogram = RGBHistogram(n_bin=8, h_type="region")
array_store = NPArrayStore(retrieve=KNNRetrieval(metric="cosine"))

cbir = CBIR(rgb_histogram, array_store)

import cv2

n1 = cv2.imread("/home/rhev/Works/Code/CV-basic/data/prj1-4/009_0.png")
n2 = cv2.imread(
    "/home/rhev/Works/Code/CV-basic/data/prj1-2/1_wIXlvBeAFtNVgJd49VObgQ.png"
)
n3 = cv2.imread("/home/rhev/Works/Code/CV-basic/data/prj1-4/001_0.png")
n4 = cv2.imread("/home/rhev/Works/Code/CV-basic/data/prj1-4/001_0.png")

cbir.indexing([n1, n2, n3])

cbir.retrieve(n4, k=1)