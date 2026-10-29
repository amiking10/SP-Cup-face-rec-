"""Visible desktop actions and screenshot feedback."""
import os
from pathlib import Path
import time
from uuid import uuid4
import cv2

SAVE_BUTTON = (10, 54, 158, 80)
FOLDER_BUTTON = (168, 54, 326, 80)


class Controls:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.save_requested = False
        self.message = ""
        self.message_until = 0.0

    def notify(self, message):
        self.message = message
        self.message_until = time.monotonic() + 6

    def click(self, x, y):
        for action, (left, top, right, bottom) in (
                (self.request_save, SAVE_BUTTON), (self.open_folder, FOLDER_BUTTON)):
            if left <= x < right and top <= y < bottom:
                action()
                return True
        return False

    def request_save(self):
        self.save_requested = True

    def open_folder(self):
        try:
            if os.name == "nt":
                os.startfile(str(self.directory))
            else:
                print(f"Results folder: {self.directory}", flush=True)
            self.notify("Results folder opened" if os.name == "nt" else str(self.directory))
        except OSError as exc:
            self.notify("Cannot open folder; see terminal")
            print(f"Cannot open results folder {self.directory}: {exc}", flush=True)

    def save(self, image, frame_index):
        self.save_requested = False
        path = self.directory / f"screenshot_{frame_index:06d}_{uuid4().hex[:8]}.png"
        try:
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise OSError("PNG encoding failed")
            # Unlike cv2.imwrite, this also handles Unicode paths on Windows.
            with path.open("xb") as output:
                output.write(encoded.tobytes())
        except (OSError, cv2.error) as exc:
            self.notify("Screenshot failed; see terminal")
            print(f"Screenshot failed: {path}: {exc}", flush=True)
            return None
        self.notify(f"Saved: {path.name}")
        print(f"Screenshot saved: {path}", flush=True)
        return path

    def draw(self, image):
        for label, (left, top, right, bottom) in (
                ("Save [S/Space]", SAVE_BUTTON), ("Open folder [O]", FOLDER_BUTTON)):
            cv2.rectangle(image, (left, top), (right, bottom), (70, 70, 70), -1)
            cv2.putText(image, label, (left + 7, top + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        if time.monotonic() < self.message_until:
            height, width = image.shape[:2]
            cv2.rectangle(image, (0, height - 34), (width, height), (25, 25, 25), -1)
            cv2.putText(image, self.message, (10, height - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
