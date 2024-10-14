import torch

from cbir.entities.search_objects import ImageSearchObject


def ensemble_search(
    *args: list[ImageSearchObject], weights: list, datalength: int, k: int = 10
) -> list[ImageSearchObject]:
    assert len(args) == len(weights), "Arguments and weights must have same length"
    
    for arg in args:
        assert isinstance(
            arg[0], ImageSearchObject
        ), "Arguments must be list of ImageSearchObject"

    scores = torch.zeros(datalength)
    for search_list, weight in zip(args, weights):
        search_scores = torch.zeros(datalength).float()
        index_tensor = torch.tensor([i.index for i in search_list])
        value_tensor = torch.tensor([s.score for s in search_list]).float()
        search_scores = search_scores.scatter_(0, index_tensor, value_tensor) * weight
        
        scores += search_scores
    
    top = scores.topk(k)
    
    return [ImageSearchObject(index, score) for index, score in zip(top.indices, top.values)]
