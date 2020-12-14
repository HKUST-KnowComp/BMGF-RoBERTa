import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# we follow the scorer of CoNLL 2015 share task
# hitting any relation in an example means correct
# otherwise, refine multiple relations into one prefered relation
# https://github.com/attapol/conll15st/blob/master/scorer.py

def evaluate_accuracy(pred, target, prefered_target=None):
    num_examples, num_classes = target.shape

    correct = 0
    pred_refined = np.zeros_like(target)
    target_refined = np.zeros_like(target)
    for i in range(num_examples):
        j = pred[i]
        pred_refined[i, j] = 1
        if target[i, j]:
            correct += 1
            target_refined[i, j] = 1
        else:
            if prefered_target is not None:
                j = prefered_target[i]
            else:
                j = target[i].argmax()
            target_refined[i, j] = 1

    # delete empty labels, map those predictions to a dummy label
    cnt = target_refined.sum(axis=0)
    real_labels = cnt > 0
    result = {}
    for c in range(num_classes):
        if real_labels[c]:
            result[c] = accuracy_score(target_refined[:, c], pred_refined[:, c])
        else:
            result[c] = 1.0
    
    result["overall"] = correct/num_examples
    return result

def evaluate_precision_recall_f1(pred, target, prefered_target=None, average="macro"):
    num_examples, num_classes = target.shape
    
    if average == "binary":
        if len(pred.shape) == 2:
            pred_refined = pred.argmax(axis=1)
        else:
            pred_refined = pred
        target_refined = target[:, 1]
        result = {"overall": tuple(precision_recall_fscore_support(target_refined, pred_refined, average="binary")[0:3])}
    else:
        pred_refined = np.zeros_like(target)
        target_refined = np.zeros_like(target)
        for i in range(num_examples):
            j = pred[i]
            pred_refined[i, j] = 1
            if target[i, j]:
                target_refined[i, j] = 1
            else:
                if prefered_target is not None:
                    j = prefered_target[i]
                else:
                    j = target[i].argmax()
                target_refined[i, j] = 1
    # delete empty labels, map those predictions to a dummy label
    cnt = target_refined.sum(axis=0)
    real_labels = cnt > 0
    result = {}
    for c in range(num_classes):
        if real_labels[c]:
            result[c] = tuple(precision_recall_fscore_support(target_refined[:, c], pred_refined[:, c], average="binary")[0:3])
        else:
            result[c] = (0.0, 0.0, 0.0)
    if pred_refined[:, cnt <= 0].sum() == 0:
        result["overall"] = tuple(precision_recall_fscore_support(target_refined[:, real_labels], pred_refined[:, real_labels], average=average)[0:3])
    else:
        result["overall"] = tuple(precision_recall_fscore_support(
            np.concatenate([target_refined[:, real_labels], np.zeros((num_examples, 1), dtype=target_refined.dtype)], axis=1),
            np.concatenate([pred_refined[:, real_labels], pred_refined[:, cnt <= 0].sum(axis=1).reshape(num_examples, 1)], axis=1),
            average=average)[0:3])
    return result