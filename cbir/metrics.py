def average_precision(predictions, ground_truths, k):
    top_k_predictions = predictions[:k]
    
    hit_count = 1
    relevant_items = []
    for i, pred in enumerate(top_k_predictions):
        if pred in ground_truths:
            relevant_items.append(hit_count/(i+1))
            hit_count += 1
    
    ap = sum(relevant_items) / hit_count
    
    return ap

def hit_rate(predictions, ground_truths, k):
    top_k_predictions = predictions[:k]
    
    relevant_items = sum(1 for pred in top_k_predictions if pred in ground_truths)
    
    # hit = 1 if relevant_items >= 1 else 0
    hit = relevant_items / len(top_k_predictions)
    
    return hit

def recall(predictions, ground_truths, k):
    top_k_predictions = predictions[:k]
    
    relevant_items = sum(1 for pred in top_k_predictions if pred in ground_truths)
    
    recall = relevant_items / len(ground_truths)
    
    return recall