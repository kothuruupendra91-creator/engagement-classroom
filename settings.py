import json

import config

DEFAULTS = {"camera_index": config.CAMERA_INDEX, "yaw_ok": config.YAW_OK, "yaw_away": config.YAW_AWAY,
            "pitch_ok": config.PITCH_OK, "pitch_away": config.PITCH_AWAY, "phone_conf": config.PHONE_CONF,
            "record_default": False, "pixelate": True}


def _path():
    return config.DB_PATH.parent / "settings.json"


def load():
    s = dict(DEFAULTS)
    try:
        s.update({k: v for k, v in json.loads(_path().read_text()).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return s


def save(s):
    _path().parent.mkdir(exist_ok=True)
    _path().write_text(json.dumps(s, indent=2))
