from metrics.iou import iou_score

def print_iou(test_batch, model):
    score = iou_score(23, model, test_batch)
    print("Test score (IoU): {:.2f}".format(score))
    return score