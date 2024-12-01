import numpy as np

# Updated confusion matrix (as provided)
confusion_matrix = np.array([
    [831,   4,  13,  42,  11,   0,  84,   0,  15,   0],
    [  4, 955,   4,  27,   5,   0,   3,   0,   2,   0],
    [ 26,   4, 716,   8, 157,   1,  79,   0,   9,   0],
    [ 33,  16,  14, 849,  44,   1,  39,   0,   4,   0],
    [  0,   1,  96,  35, 781,   1,  77,   0,   9,   0],
    [  1,   0,   0,   1,   0, 884,   0,  58,   9,  47],
    [152,   1, 116,  34, 114,   1, 555,   0,  27,   0],
    [  0,   0,   0,   0,   0,  29,   0, 929,   0,  42],
    [  3,   1,   5,   9,   2,   3,  24,   6, 946,   1],
    [  0,   0,   0,   0,   0,   5,   0,  39,   1, 955]
])

def calculate_metrics(confusion_matrix):
    """
    Calculate precision, recall, F1 score, and F2 score for each class.
    """
    num_classes = confusion_matrix.shape[0]
    metrics = []

    for i in range(num_classes):
        TP = confusion_matrix[i, i]  # True Positives
        FP = np.sum(confusion_matrix[:, i]) - TP  # False Positives
        FN = np.sum(confusion_matrix[i, :]) - TP  # False Negatives
        TN = np.sum(confusion_matrix) - (TP + FP + FN)  # True Negatives (not used here)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f2_score = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
        
        metrics.append((precision, recall, f1_score, f2_score))
    
    return metrics

# Calculate metrics
metrics = calculate_metrics(confusion_matrix)

# Display metrics
print("Metrics for each class:")
for i, (precision, recall, f1, f2) in enumerate(metrics):
    print(f"Class {i}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  F2 Score: {f2:.4f}")
    print()
