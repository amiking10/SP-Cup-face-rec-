"""Tiny generated-video fixtures test infrastructure, not face tracking quality."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from c3_tracker.tracking import Observation
from c3_tracker.benchmark.schema import GroundTruth, VideoSpec, Sequence, GT_FIELDS, write_csv
from c3_tracker.benchmark.inference import PredictedFrame


def synthetic_ava(tmp_path, fields=8, coverage='complete'):
    raw = tmp_path/'raw';raw.mkdir()
    video = raw/'toy.avi'
    writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'FFV1'),10,(100,64))
    assert writer.isOpened()
    rows=[]
    for index in range(8):
        image=np.zeros((64,100,3),dtype=np.uint8)
        boxes=[('a',(10+index,20,30+index,40),(0,0,255)),('b',(70-index,20,90-index,40),(0,255,0))]
        for person,box,color in boxes:
            x1,y1,x2,y2=box;image[y1:y2,x1:x2]=color
            coordinates=[x1/100,y1/64,x2/100,y2/64]
            row=['toy',str(900+index/10),*coordinates,'NOT_SPEAKING',person]
            if fields==5:
                row=['toy',str(900+index/10),str(tuple(coordinates)),'NOT_SPEAKING',person]
            rows.append(row)
        writer.write(image)
    writer.release()
    with (raw/'toy.csv').open('w',newline='') as f: csv.writer(f).writerows(rows)
    payload={'dataset':'ava','release':'synthetic-not-a-dataset-score','split':'synthetic','videos':[{
        'video_id':'toy','video_path':'toy.avi','annotation_path':'toy.csv','video_start_timestamp_s':900,
        'cfr':True,'coverage':coverage,'identity_annotations_complete':True,
        'coverage_provenance':'All objects and boxes generated exhaustively by the synthetic fixture'}]}
    manifest=raw/'manifest.json';manifest.write_text(json.dumps(payload))
    return manifest


class ColorBackend:
    """Tests only. No GT access: extracts boxes by colors from decoded pixels."""
    def __init__(self,config):
        self.config=config;self.designations=[];self.oracle_calls=[];self.closed=False

    def process(self,frame,timestamp_s,oracle_boxes=None):
        self.oracle_calls.append(oracle_boxes)
        if self.config.experiment=='oracle':
            assert oracle_boxes is not None
            assert all(isinstance(box,tuple) and len(box)==4 for box in oracle_boxes)
            return [Observation(str(i),tuple(box),True,True) for i,box in enumerate(oracle_boxes)]
        assert oracle_boxes is None
        observations=[]
        for name,channel in [('red',2),('green',1)]:
            mask=(frame[:,:,channel]>150).astype(np.uint8)
            points=cv2.findNonZero(mask)
            if points is not None:
                x,y,w,h=cv2.boundingRect(points)
                observations.append(Observation(name,(x,y,x+w,y+h),True,True))
        return observations

    def designate(self,track_id): self.designations.append(track_id)
    def close(self): self.closed=True


def metric_sequence(tmp_path, ground_truth, scope='persistent', coverage='complete'):
    video=VideoSpec('toy',tmp_path/'unused.avi',100,64,10,20,0)
    metadata={'coverage':coverage,'identity_scope':scope,'identity_annotations_complete':True,
              'absence_annotations_complete':scope=='persistent','identity_provenance':'synthetic known physical identities'}
    return Sequence(video,ground_truth,metadata,tmp_path/'unused.csv')


def gt(index,person='a',box=(10,20,30,40),visible=True):
    return GroundTruth('toy',index,index/10,person,box if visible else None,visible)


def prediction(index,observations,selected='1',status=None):
    if status is None:
        status='VISIBLE' if any(o.track_id==selected and o.observed and o.confirmed for o in observations) else 'LOST'
    return PredictedFrame(index,index,index/10,tuple(observations),1.0,selected,status,100,64)


def obs(ident='1',box=(10,20,30,40),observed=True,confirmed=True):
    return Observation(ident,box,observed,confirmed)


def audit(frame=0,person='a',track='1',status='initialized'):
    return {'status':status,'person_id':person,'selected_track_id':track,'frame_index':frame}
