import os
from pathlib import Path as P
from configparser import ConfigParser


CONF_PATH = "config.txt"


def get_param_from_config():
    conf = ConfigParser()
    conf.read(CONF_PATH)
    z = conf["DEFAULT"].get("z", 0)
    c = conf["DEFAULT"].get("channel", 0)
    t = conf["DEFAULT"].get("time", 0)

    return f" -z {z} -c {c} -t {t}"


if __name__ == "__main__":
    args = get_param_from_config()
    dn = "ProcessingBox"
    for f in os.listdir(dn):
        fullpath = P(dn) / f
        if not fullpath.is_file():
            continue
        if not (f.endswith(".tif") or f.endswith(".tiff")):
            continue

        print("Segmenting file :", f)
        cmd1 = f"python3 src/script_maskrcnn_v2.py -f {fullpath} {args}"
        cmd2 = f"python3 src/script_gather_roi.py -f {fullpath}"
        print(f"Running {cmd1}")
        os.system(cmd1)
        print(f"Running {cmd2}")
        os.system(cmd2)
