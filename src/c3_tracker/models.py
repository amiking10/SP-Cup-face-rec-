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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', required=True)
    parser.add_argument('--model-dir', type=Path, default=Path('.models'))
    args = parser.parse_args()
    download(args.model_dir)
