from torchmetrics import JaccardIndex
import torch
import numpy as np
import torch.nn as nn

def iou_score(n_classes, model, loader):
    iou = JaccardIndex(num_classes=n_classes)
    iou_scores = [] 
    with torch.no_grad():
            for x, y in loader:
                x = x.to('cuda')
                softmax = nn.Softmax(dim=1)
                preds = torch.argmax(softmax(model(x)),axis=1).to('cpu')
                score = np.array(iou(preds, y))
                iou_scores.append(score)
    total_score = np.sum(np.array(iou_scores*100))
    return total_score/len(iou_scores)