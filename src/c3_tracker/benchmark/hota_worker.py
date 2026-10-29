"""Standalone TrackEval bridge. Run with a separate NumPy-2 metrics interpreter."""
import json
from pathlib import Path
import sys
import platform
from importlib.metadata import version
import numpy as np
from trackeval.metrics import HOTA


def calculate(payload):
    metric = HOTA()
    all_results, per_video = {}, {}
    for video, blocks in payload['groups'].items():
        results = {}
        for index, block in enumerate(blocks):
            block = dict(block)
            block['gt_ids'] = [np.asarray(ids, dtype=int) for ids in block['gt_ids']]
            block['tracker_ids'] = [np.asarray(ids, dtype=int) for ids in block['tracker_ids']]
            block['similarity_scores'] = [np.asarray(scores, dtype=float).reshape(len(gt), len(pred))
                                          for scores,gt,pred in zip(block['similarity_scores'], block['gt_ids'], block['tracker_ids'])]
            results[str(index)] = metric.eval_sequence(block)
        if results:
            per_video[video] = metric.combine_sequences(results)
            all_results.update({f'{video}:{index}': result for index,result in results.items()})
    def summary(result):
        return {name.lower(): float(np.mean(result[name])) for name in ('HOTA','DetA','AssA','LocA')}
    return {'library': 'TrackEval', 'version': version('trackeval'),
            'python': platform.python_version(), 'numpy': np.__version__, 'scipy': version('scipy'),
            'thresholds': metric.array_labels.tolist(),
            'per_video': {key: summary(value) for key,value in per_video.items()},
            'overall': summary(metric.combine_sequences(all_results)) if all_results else None}


if __name__ == '__main__':
    payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    Path(sys.argv[2]).write_text(json.dumps(calculate(payload), indent=2, allow_nan=False), encoding='utf-8')
