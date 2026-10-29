"""Explicit download and integrity checks for the OpenCV face models."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

ZOO_COMMIT = '48b10fa505aeb51fbd127fc60a5c660040e823b7'
MODELS = {
    'face_detection_yunet_2023mar.onnx': (
        'face_detection_yunet', '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'),
    'face_recognition_sface_2021dec.onnx': (
        'face_recognition_sface', '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'),
}


def model_path(name, directory=None):
    path = Path(directory or '.models') / name
    if not path.is_file():
        raise RuntimeError(f'Missing face model: {path}. Run: python -m c3_tracker.models --download')
    with path.open('rb') as model:
        digest = hashlib.file_digest(model, 'sha256').hexdigest()
    if digest != MODELS[name][1]:
        raise RuntimeError(f'Face model integrity check failed: {path}')
    return path


def download(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, (folder, digest) in MODELS.items():
        path = directory / name
        if path.is_file():
            model_path(name, directory)
            continue
        url = f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{ZOO_COMMIT}/models/{folder}/{name}'
        print(f'Downloading {name} from OpenCV Zoo', flush=True)
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f'Download integrity check failed: {name}')
        path.write_bytes(data)
        print(f'Installed: {path.resolve()}', flush=True)


def download_test_fixture(directory):
    path = Path(directory) / 'testdata' / 'other_face.jpg'
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = '7de7ed51a1594fff247f4cae2301eceacf5313d6011e37b4a4c8733f7bb72c07'
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return
    with urlopen('https://raw.githubusercontent.com/opencv/opencv/4.11.0/samples/data/lena.jpg', timeout=30) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError('OpenCV test fixture integrity check failed')
    path.write_bytes(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', required=True)
    parser.add_argument('--model-dir', type=Path, default=Path('.models'))
    parser.add_argument('--test-fixtures', action='store_true', help='also fetch the public OpenCV sample for model regressions')
    args = parser.parse_args()
    download(args.model_dir)
    if args.test_fixtures:
        download_test_fixture(args.model_dir)
