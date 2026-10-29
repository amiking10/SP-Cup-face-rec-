"""Frame-local one-to-one IoU matching; no identity labels influence costs."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(first, second):
    result = np.zeros((len(first), len(second)), dtype=float)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            union = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - intersection
            result[i, j] = intersection / union if union > 0 else 0
    return result


def match_boxes(first, second, threshold=.5):
    overlaps = iou_matrix(first, second)
    if not first or not second:
        return [], list(range(len(first))), list(range(len(second))), overlaps
    # Large invalid cost maximizes valid match cardinality before IoU sum.
    costs = np.where(overlaps >= threshold, 1 - overlaps, 1e6)
    indices = zip(*linear_sum_assignment(costs))
    matches = [(int(i), int(j)) for i, j in indices if overlaps[i, j] >= threshold]
    return (matches, [i for i in range(len(first)) if i not in {a for a, _ in matches}],
            [j for j in range(len(second)) if j not in {b for _, b in matches}], overlaps)
