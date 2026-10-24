from collections import Counter, deque

import config


def _info(s):
    if s["bbox"] is not None:
        x0, y0, x1, y1 = s["bbox"]
        return (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, 40.0)
    return s["nose"][0], s["nose"][1], 80.0


def _majority(hist):
    c = Counter(hist)
    top = max(c.values())
    for v in reversed(hist):          # on a tie, the most recent value wins
        if c[v] == top:
            return v


class StudentTracker:
    """Keeps each student's state steady from frame to frame.

    Only positions are remembered in memory (S1, S2, ...). No faces are stored or recognised.
    """

    def __init__(self):
        self.tracks, self.frame, self.next_id = [], 0, 1

    def _touch(self, t, s, info):
        t["cx"], t["cy"], t["w"] = info
        t["bbox"], t["nose"] = s["bbox"], s["nose"]
        t["dyaw"], t["dpitch"] = s["dyaw"], s["dpitch"]
        t["hist_head"].append(s["head"])
        t["hist_hand"].append(bool(s["hand"]))
        t["hist_phone"].append(bool(s["phone"]))
        t["hist_emo"].append((s.get("emotion"), s.get("emo_conf", 0.0)))
        t["seen"] = self.frame
        t["hits"] += 1

    def update(self, students):
        self.frame += 1
        infos = [_info(s) for s in students]
        pairs = []
        for ti, t in enumerate(self.tracks):
            for si, (cx, cy, w) in enumerate(infos):
                d = ((t["cx"] - cx) ** 2 + (t["cy"] - cy) ** 2) ** 0.5
                if d <= max(60.0, 1.5 * max(w, t["w"])):
                    pairs.append((d, ti, si))
        pairs.sort()
        used_t, used_s = set(), set()
        for _, ti, si in pairs:
            if ti in used_t or si in used_s:
                continue
            used_t.add(ti)
            used_s.add(si)
            self._touch(self.tracks[ti], students[si], infos[si])
        for si, s in enumerate(students):
            if si not in used_s:
                n = config.STABLE_FRAMES
                t = {"id": self.next_id, "hits": 0, "hist_head": deque(maxlen=n),
                     "hist_hand": deque(maxlen=n), "hist_phone": deque(maxlen=n),
                     "hist_emo": deque(maxlen=n)}
                self.next_id += 1
                self._touch(t, s, infos[si])
                self.tracks.append(t)
        self.tracks = [t for t in self.tracks if self.frame - t["seen"] <= config.TRACK_FORGET_FRAMES]

        out = []
        for t in self.tracks:
            missed = self.frame - t["seen"]
            if t["hits"] < config.TRACK_MIN_HITS or missed > config.TRACK_HOLD_FRAMES:
                continue
            hand = sum(t["hist_hand"]) / len(t["hist_hand"]) >= config.HAND_HOLD_RATIO
            phone = sum(t["hist_phone"]) / len(t["hist_phone"]) >= config.PHONE_HOLD_RATIO
            labels = [e for e, _ in t["hist_emo"] if e]
            emo, conf = None, 0.0
            if len(labels) >= config.TRACK_MIN_HITS:
                emo = _majority(labels)
                cs = [c for e, c in t["hist_emo"] if e == emo]
                conf = sum(cs) / len(cs)
            out.append({"emotion": emo, "emo_conf": conf, "id": t["id"], "bbox": t["bbox"], "nose": t["nose"],
                        "hidden": t["bbox"] is None, "head": _majority(t["hist_head"]),
                        "hand": hand, "phone": phone,
                        "dyaw": t["dyaw"], "dpitch": t["dpitch"]})
        return out
