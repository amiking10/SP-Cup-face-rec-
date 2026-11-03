"""Lossless paired exports from tracking logs; no reference boxes enter crop generation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2

from .paired_export import digest
from ..paired_recording import PairedRecorder
from ..roi import capture_frame


def tracking_frames(record):
    with (Path(record['log_dir'])/'tracking_log.csv').open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    groups={}
    for row in rows: groups.setdefault(int(row['frame_index']),[]).append(row)
    if sorted(groups)!=list(range(record['source_frames'])): raise ValueError('Tracking log skips source frames')
    out={}
    for index,group in groups.items():
        visible={r['target_visible'] for r in group}
        if len(visible)!=1: raise ValueError('Conflicting target visibility')
        current=[r for r in group if r['is_target']=='1' and r['has_detection']=='1' and r['confirmed']=='1']
        if len(current)>1 or (visible=={'1'} and len(current)!=1): raise ValueError('Invalid current target observation')
        row=current[0] if current else None
        roi=tuple(int(float(row[f'capture_roi_{s}'])) for s in ('left','top','right','bottom')) if row is not None else None
        box=tuple(float(row[s]) for s in ('left','top','right','bottom')) if row is not None else None
        out[index]={'visible':visible=={'1'},'roi':roi,'box':box}
    return out


def render_pair(export_path,directory,start_frame=30):
    record=json.loads(Path(export_path).read_text(encoding='utf-8'))
    if record['status'] not in ('complete','no_selection'): raise ValueError('Tracking did not complete')
    if record['source_frames_processed']!=record['source_frames']: raise ValueError('Incomplete tracking source')
    source=Path(record['source_video'])
    if digest(source)!=record['source_sha256']: raise ValueError('Source video changed after tracking')
    frames=tracking_frames(record);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS)
    width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not 0<=start_frame<record['source_frames']: raise ValueError('Invalid paired start')
    first=frames[start_frame]
    pair=PairedRecorder(directory,fps,source,(width,height),dict(top=0,bottom=0,left=0,right=0),
                        loss_policy='mark',video_format='lossless')
    selected=record.get('target_id');tag=selected or 'unselected'
    pair.metadata.update(initialization_status='selected' if selected else 'failed',raw_selected_track_id=selected,
                         benchmark='Synthetic composite; one-frame designated target; fixed timeline',
                         unselected_label='unselected is an abstention marker, never an assigned raw track')
    pair.begin(tag,first['roi'] or (0,0,min(width,128),min(height,128)))
    hashes={name:hashlib.sha256() for name in pair.video_files.values()}
    index=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if index>=start_frame:
                state=frames[index]
                pair.update(frame,index,tag,state['visible'],state['roi'],cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                original=cv2.copyMakeBorder(frame,0,height%2,0,width%2,cv2.BORDER_CONSTANT,value=(0,0,0))
                cropped=capture_frame(frame,state['roi'] if state['visible'] else None,pair.crop_size)
                hashes[pair.video_files['original']].update(original.tobytes())
                hashes[pair.video_files['cropped']].update(cropped.tobytes())
            index+=1
        if index!=record['source_frames']: raise ValueError('Unexpected composite length')
        pair.metadata['expected_pixel_sha256']={name:h.hexdigest() for name,h in hashes.items()}
        pair.finish('end_of_video')
    finally:
        cap.release()
        if pair.status=='recording':pair.finish('processing_failed')
    return pair.metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracking-export',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start-frame',type=int,default=30)
    args=parser.parse_args();render_pair(args.tracking_export,args.output,args.start_frame)


if __name__=='__main__':main()
