import pytest
from c3_tracker.app import parser, open_capture


def test_missing_video(tmp_path):
    args = parser().parse_args(['--source', 'video', '--input', str(tmp_path / 'missing.mp4')])
    with pytest.raises(ValueError, match='existing file'):
        open_capture(args)


def test_negative_camera():
    args = parser().parse_args(['--source', 'webcam', '--camera-index', '-1'])
    with pytest.raises(ValueError, match='nonnegative'):
        open_capture(args)


def test_unavailable_camera(monkeypatch):
    class Closed:
        released = False

        def set(self, *args):
            pass

        def isOpened(self):
            return False

        def release(self):
            self.released = True
    cap = Closed()
    monkeypatch.setattr('c3_tracker.app.cv2.VideoCapture', lambda *_: cap)
    with pytest.raises(RuntimeError, match='Cannot open input'):
        open_capture(parser().parse_args(['--source', 'webcam']))
    assert cap.released


@pytest.mark.parametrize('args', [[], ['--source', 'url'], ['--source', 'webcam', '--width', '0'],
                                  ['--source', 'webcam', '--confidence', '1.1']])
def test_invalid_cli(args):
    with pytest.raises(SystemExit):
        parser().parse_args(args)


def test_corrupt_video(tmp_path):
    path = tmp_path / 'bad.mp4'
    path.write_text('not video')
    with pytest.raises(RuntimeError, match='Cannot open input'):
        open_capture(parser().parse_args(['--source', 'video', '--input', str(path)]))
