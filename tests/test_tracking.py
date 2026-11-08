from c3_tracker.tracking import observations_from_tracks


class MockTrack:
    track_id = 17
    time_since_update = 1

    def to_ltrb(self, orig=False, orig_strict=False):
        return None if orig and orig_strict else (-5, 0, 20, 30)

    def is_confirmed(self):
        return True

    def get_det_conf(self):
        raise AssertionError('predicted track has no detection confidence')


def test_prediction_does_not_count_as_observation():
    o, = observations_from_tracks([MockTrack()], 100, 100)
    assert o.bbox == (0, 0, 20, 30)
    assert o.track_id == '17'
    assert not o.observed
    assert o.detection_confidence is None


def test_current_original_box_used():
    track = MockTrack()
    track.time_since_update = 0
    track.to_ltrb = lambda **kwargs: (10, 20, 30, 40) if kwargs.get('orig') else (0, 0, 40, 50)
    track.get_det_conf = lambda: .8
    o, = observations_from_tracks([track], 100, 100)
    assert o.observed and o.confirmed
    assert o.bbox == (10, 20, 30, 40)
    assert o.detection_confidence == .8


def test_embedding_reconfigures_yunet_only_when_frame_size_changes(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from c3_tracker import embedding
    from c3_tracker.detection import Detection
    sizes=[]
    detection=Detection((0,0,20,20),.9)
    face=np.array([0,0,20,20],np.float32)
    detector=SimpleNamespace(setInputSize=lambda size:sizes.append(size),
                             detect=lambda frame:(None,np.array([face])))
    recognizer=SimpleNamespace(alignCrop=lambda frame,box:frame,
                               feature=lambda crop:np.array([[3.,4.]],np.float32))
    monkeypatch.setattr(embedding,'model_path',lambda *a:'unused.onnx')
    monkeypatch.setattr(embedding.cv2,'FaceDetectorYN',SimpleNamespace(create=lambda *a:detector))
    monkeypatch.setattr(embedding.cv2,'FaceRecognizerSF',SimpleNamespace(create=lambda *a:recognizer))
    embedder=embedding.FaceEmbedder()
    for width,height in [(320,320),(640,480),(640,480),(320,320),(320,320)]:
        accepted,features=embedder.encode([detection],np.zeros((height,width,3),np.uint8))
        assert accepted==[detection]
        np.testing.assert_allclose(features,[[.6,.8]])
    assert sizes==[(640,480),(320,320)]
    assert embedder.encode([],np.zeros((720,1280,3),np.uint8))==([],[])
    assert sizes==[(640,480),(320,320)]
