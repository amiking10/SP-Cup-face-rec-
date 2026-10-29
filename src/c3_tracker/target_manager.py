"""No automatic reassignment or cross-ID recovery."""


class TargetManager:
    def __init__(self):
        self.reset()

    def reset(self):
        self.target_id = None
        self.status = "NOT SELECTED"
        self._was_lost = False

    def select(self, x, y, observations):
        candidates = [o for o in observations if o.confirmed and o.observed
                      and o.bbox[0] <= x < o.bbox[2] and o.bbox[1] <= y < o.bbox[3]]
        if not candidates:
            return False
        # In overlap select the smallest enclosing box; ID breaks ties deterministically.
        selected = min(candidates, key=lambda o: (
            (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]), o.track_id))
        self.target_id = selected.track_id
        self.status = "VISIBLE"
        self._was_lost = False
        return True

    def update(self, observations):
        if self.target_id is None:
            self.status = "NOT SELECTED"
            return self.status
        visible = any(o.track_id == self.target_id and o.confirmed and o.observed
                      for o in observations)
        if visible:
            self.status = "REACQUIRED" if self._was_lost else "VISIBLE"
            self._was_lost = False
        else:
            self.status = "LOST"
            self._was_lost = True
        return self.status

    @property
    def visible(self):
        return self.status in ("VISIBLE", "REACQUIRED")
