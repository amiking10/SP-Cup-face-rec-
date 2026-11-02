"""Headless single-face dataset export using the desktop tracking and ROI code."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np

from .backend import ExistingTrackerBackend
from .inference import BenchmarkConfig
from ..logging_utils import RunLogger, write_json
from ..paired_recording import PairedRecorder
from ..roi import selected_capture_roi
from ..target_manager import TargetManager


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def export_video(source,directory,subject,run_id,config=None,backend_factory=ExistingTrackerBackend):
    source=Path(source).resolve();directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'export.json').exists(): raise ValueError('Export already exists; use a new attempt directory')
    config=config or BenchmarkConfig()
    random.seed(config.seed);np.random.seed(config.seed);cv2.setRNGSeed(config.seed);cv2.setNumThreads(config.threads)
    cap=cv2.VideoCapture(str(source));backend=None;logger=None;pair=None
    index=0;selected_at=None;started=time.monotonic()
    record={'run_id':run_id,'subject':subject,'source_video':str(source),'configuration':asdict(config),
            'status':'INVALID','selection_policy':'First confirmed observed face; largest area then ID; never replace selected ID; no ground truth',
            'source_sha256':digest(source)}
    try:
        ok,frame=cap.read()
        if not ok: raise RuntimeError('Cannot decode source video')
        fps=float(cap.get(cv2.CAP_PROP_FPS));height,width=frame.shape[:2]
        expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        record.update(source_fps=fps,source_frames=expected,source_resolution=[width,height])
        logger=RunLogger(directory,record)
        pair=PairedRecorder(logger.directory,fps,source,(width,height),dict(top=0,bottom=0,left=0,right=0),loss_policy='mark')
        backend=backend_factory(config);target=TargetManager()
        while ok:
            tick=time.monotonic();observations=backend.process(frame,index/fps)
            if target.target_id is None:
                candidates=[o for o in observations if o.observed and o.confirmed]
                if candidates:
                    chosen=min(candidates,key=lambda o:(-(o.bbox[2]-o.bbox[0])*(o.bbox[3]-o.bbox[1]),o.track_id))
                    target.target_id=chosen.track_id;backend.designate(chosen.track_id);selected_at=index
            target.update(observations)
            roi=selected_capture_roi(observations,target,width,height)
            pair.update(frame,index,target.target_id,target.visible,roi,cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
            logger.log_frame(index,index/fps,observations,target,(time.monotonic()-tick)*1000,capture_roi=roi)
            index+=1
            if index%300==0: print(subject,index,'/',expected,flush=True)
            ok,frame=cap.read()
        if index!=expected: raise RuntimeError(f'Unexpected source frame count {index} != {expected}')
        pair.finish('end_of_video')
        record.update(status='complete' if pair.count else 'no_selection',pair_dir=str(logger.directory),target_id=target.target_id)
    except BaseException as exc:
        record['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        cap.release()
        if backend is not None: backend.close()
        try:
            if pair is not None: pair.finish('processing_failed')
        finally:
            record.update(source_frames_processed=index,selected_at_source_frame=selected_at,elapsed_s=time.monotonic()-started)
            if logger is not None: logger.close(record)
            write_json(directory/'export.json',record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--subject',required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    result=export_video(args.source,args.output,args.subject,args.run_id)
    print(json.dumps(result),flush=True)
    if result['status']!='complete': raise SystemExit(2)


if __name__=='__main__': main()
