# Content-based Image Retrieval

### Download and Split Data
We use caltech101 dataset to evaluate our system. The dataset can be downloaded from [here](https://www.kaggle.com/datasets/imbikramsaha/caltech-101).

To split train, test and valid set. Please do the following step:
* Extract zip file into `data/caltech-101`
* Install `split-folders` python package 
```shell
pip install split-folders
```
* Run this python scripts
```python
import splitfolders
splitfolders.ratio('data/caltech-101', output="data/caltech101", seed=1337, ratio=(0.7, 0.15,0.15)) 
```

### Reproduce result
* All results are stored in `out/` directory

* Install dependencies: `pip install -r requirements.txt`

* Run `bash.sh` to run evaluation on 4 methods of feature extraction: color histogram, sift description, resnet and ensemble.