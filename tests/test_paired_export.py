import csv
import json
import cv2
import numpy as np
from c3_tracker.tracking import Observation


def source_video(tmp_path):
    path=tmp_path/'base.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),5,(64,48))
    assert writer.isOpened()
    for i in range(15): writer.write(np.full((48,64,3),90+i,np.uint8))
    writer.release()
    return path


def test_export_keeps_loss_frames_and_never_reselects_other_id(tmp_path):
    from c3_tracker.benchmark.paired_export import export_video
    class Backend:
        def __init__(self,config): self.selected=[]
        def process(self,frame,timestamp_s):
            i=round(timestamp_s*5)
            if i<2: return [Observation('1',(10,10,30,30),True,False)]
            if 5<=i<9: return [Observation('2',(20,10,40,30),True,True)]
            return [Observation('1',(10,10,30,30),True,True)]
        def designate(self,track_id): assert track_id=='1';self.selected.append(track_id)
        def close(self): pass
    result=export_video(source_video(tmp_path),tmp_path/'out','subject1','synthetic',backend_factory=Backend)
    assert result['status']=='complete' and result['selected_at_source_frame']==2
    meta=json.loads((__import__('pathlib').Path(result['pair_dir'])/'paired_recording.json').read_text())
    assert meta['frames_written']==13 and meta['lost_face_frames']==4 and meta['selected_track_id']=='1'


def test_unselected_subject_is_an_explicit_failure_not_a_missing_result(tmp_path):
    from c3_tracker.benchmark.paired_export import export_video
    class Backend:
        def __init__(self,config): pass
        def process(self,*args): return []
        def close(self): pass
    result=export_video(source_video(tmp_path),tmp_path/'out','subject1','synthetic',backend_factory=Backend)
    assert result['status']=='no_selection' and result['source_frames_processed']==15


def test_designated_click_does_not_choose_larger_bystander_or_retry(tmp_path):
    from c3_tracker.benchmark.paired_export import export_video
    class Backend:
        def __init__(self,config): pass
        def process(self,frame,timestamp_s):
            return [Observation('1',(5,5,25,25),True,True),Observation('2',(30,0,64,48),True,True)]
        def designate(self,track_id): assert track_id=='1'
        def close(self): pass
    result=export_video(source_video(tmp_path),tmp_path/'out','synthetic','synthetic',backend_factory=Backend,
                        selection={'frame_index':3,'bbox':[5,5,25,25]},record_pair=False)
    assert result['target_id']=='1' and result['selected_at_source_frame']==3
    assert result['pair_dir'] is None and result['log_dir']
    class Unconfirmed(Backend):
        def process(self,frame,timestamp_s):
            return [Observation('1',(5,5,25,25),True,timestamp_s>.6)]
        def designate(self,track_id): raise AssertionError('Unexpected retry')
    result=export_video(source_video(tmp_path),tmp_path/'missing','synthetic','synthetic',backend_factory=Unconfirmed,
                        selection={'frame_index':3,'bbox':[5,5,25,25]},record_pair=False)
    assert result['status']=='no_selection'
    from c3_tracker.benchmark.composite_pair import render_pair
    metadata=render_pair(tmp_path/'missing/export.json',tmp_path/'missing_pair',start_frame=3)
    assert metadata['frames_written']==12 and metadata['lost_face_frames']==12
    assert metadata['initialization_status']=='failed' and metadata['raw_selected_track_id'] is None
    assert metadata['verification']['original.mkv']['pixel_sha256']==metadata['expected_pixel_sha256']['original.mkv']
