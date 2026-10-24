import config

PHONE_CLASS_ID = 67 # "cell phone" in the COCO dataset


class PhoneDetector:
    def __init__(self):
        from ultralytics import YOLO # imported here so the app still runs without it
        self.model = YOLO(config.PHONE_MODEL) # downloads the weights on first use
        if self.model.names.get(PHONE_CLASS_ID) != "cell phone":
            raise RuntimeError("This YOLO model does not have a 'cell phone' class at id 67")

    def detect(self, frame_bgr):
        """Return a list of (x0, y0, x1, y1, confidence)."""
        r = self.model.predict(frame_bgr, classes=[PHONE_CLASS_ID], conf=config.PHONE_CONF,
                               imgsz=config.PHONE_IMGSZ, verbose=False)[0]
        out = []
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)))
        return out


def _box(s):
    if s["bbox"] is not None:
        return s["bbox"]
    x, y = s["nose"]
    return (x - 60, y - 60, x + 60, y + 60)


def assign_phones(students, phone_boxes):
    """Give each phone to the closest student; if none is near, still use the nearest one."""
    for pb in phone_boxes:
        cx, cy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
        best, best_d = None, 1e18 # nearest student whose area contains the phone
        near, near_d = None, 1e18 # nearest student overall
        for i, s in enumerate(students):
            x0, y0, x1, y1 = _box(s)
            w, h = x1 - x0, y1 - y0
            d = ((x0 + x1) / 2 - cx) ** 2 + ((y0 + y1) / 2 - cy) ** 2
            if d < near_d:
                near, near_d = i, d
            if x0 - 1.5 * w <= cx <= x1 + 1.5 * w and y0 - 0.5 * h <= cy <= y1 + 3 * h and d < best_d:
                best, best_d = i, d
        pick = best if best is not None else (near if config.PHONE_ASSIGN_ANY else None)
        if pick is not None:
            students[pick]["phone"] = True
