import csv
import json

import numpy as np

from c3_tracker.paired_recording import PairedRecorder
from test_paired_recording import decode


def test_mark_keeps_every_frame_and_returns_to_valid_crop(tmp_path):
    pair = PairedRecorder(tmp_path, 25, tmp_path/'base.mp4', (64, 48), {}, loss_policy='mark')
    frame = np.full((48, 64, 3), 150, np.uint8)
    for index in range(8):
        visible = index not in (0, 3, 4)
        pair.update(frame, index, '1' if index else None, visible,
                    (4, 4, 60, 44) if visible else None, index/25)
    pair.finish('end_of_input')
    original, fps = decode(tmp_path/'original.mp4')
    cropped, crop_fps = decode(tmp_path/'face_centered.mp4')
    assert len(original) == len(cropped) == 7 and fps == crop_fps == 25
    assert all(image.mean() > 140 for image in original)
    assert cropped[2].mean() < 60 and cropped[3].mean() < 60
    assert cropped[4].mean() > 140
    with (tmp_path/'frame_validity.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['source_frame_index']) for row in rows] == list(range(1, 8))
    assert [row['face_valid'] for row in rows] == ['1','1','0','0','1','1','1']
    assert [float(row['timestamp_s']) for row in rows] == [i/25 for i in range(7)]
    meta = json.loads((tmp_path/'paired_recording.json').read_text())
    assert meta['lost_face_frames'] == 2 and meta['valid_face_frames'] == 5
    assert meta['stop_reason'] == 'end_of_input'


def test_mark_still_stops_on_target_change(tmp_path):
    pair = PairedRecorder(tmp_path, 25, tmp_path/'base.mp4', (64, 48), {}, loss_policy='mark')
    frame = np.full((48,64,3),150,np.uint8)
    pair.update(frame,0,'1',True,(4,4,60,44),0)
    pair.update(frame,1,'2',True,(4,4,60,44),1/25)
    assert pair.status == 'saved' and pair.count == 1


def test_lossless_exports_preserve_bgr_pixels_and_declared_filenames(tmp_path):
    from c3_tracker.roi import capture_frame
    pair=PairedRecorder(tmp_path,30,tmp_path/'base.mkv',(64,48),{},loss_policy='mark',video_format='lossless')
    frame=np.random.default_rng(0).integers(0,256,(48,64,3),dtype=np.uint8)
    roi=(4,4,60,44)
    pair.begin('1',roi)
    pair.update(frame,30,'1',True,roi,1)
    pair.update(frame,31,'1',False,None,31/30)
    pair.finish('end_of_input')
    original,_=decode(tmp_path/'original.mkv');cropped,_=decode(tmp_path/'face_centered.mkv')
    assert np.array_equal(original[0],frame) and np.array_equal(original[1],frame)
    assert np.array_equal(cropped[0],capture_frame(frame,roi,(56,40)))
    assert np.array_equal(cropped[1],capture_frame(frame,None,(56,40)))
    assert pair.metadata['video_files']=={'original':'original.mkv','cropped':'face_centered.mkv'}
