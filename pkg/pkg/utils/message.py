from pathlib import Path

import requests

current_path = Path(__file__).parent
with open(current_path / "hook.txt", "r") as f:
    URL = f.read().strip()


def send_message(content):
    out = dict(content=content)
    try:
        requests.post(URL, json=out)
    except:
        try:
            requests.post(URL, json=out)
        except:
            pass
        pass
