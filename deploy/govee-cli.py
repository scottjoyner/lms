#!/usr/bin/env python3
"""govee.py — fleet smart-plug power control (Govee legacy API).

Safety: these plugs power infrastructure (NAS/router, GPU towers, pi DNS).
A `turn off` cuts hard power to the attached equipment. Use deliberately.

Usage:
  govee.py list
  govee.py state [name]
  govee.py on <name>
  govee.py off <name>

Names (case-insensitive substring): macbook, eenas|router, towers|r9700xt, pi|dns
"""
import json, os, sys, urllib.request

KEY_PATH = os.path.expanduser("~/.config/govee/api-key")
BASE = "https://developer-api.govee.com/v1"
ALIASES = {
    "macbook": ["macbook"],
    "router": ["eenas", "x1-370", "router"],
    "towers": ["towers", "r9700xt"],
    "dns": ["pi dns", "pi", "dns"],
}

def key():
    with open(KEY_PATH) as f:
        k = f.read().strip()
    if not k:
        sys.exit(f"empty key at {KEY_PATH}")
    return k

def api(method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method,
        headers={"Govee-API-Key": key(), "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def devices():
    return api("GET", "/devices")["data"]["devices"]

def match(query):
    q = query.lower()
    hits = []
    for d in devices():
        hay = (d["deviceName"] + " " + d["model"]).lower()
        terms = ALIASES.get(q, [q])
        if any(t in hay for t in terms):
            hits.append(d)
    if not hits:
        sys.exit(f"no device matches {query!r}")
    return hits[0]

def show_state(dev):
    d = api("GET", "/devices/state", None) if False else state_of(dev)
    props = {k: v for p in d["properties"] for k, v in p.items()}
    print(f"{dev['deviceName']:28s} power={props.get('powerState')}")

def state_of(dev):
    from urllib.parse import urlencode
    url = f"{BASE}/devices/state?" + urlencode({"device": dev["device"], "model": dev["model"]})
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Govee-API-Key": key()}), timeout=15) as r:
        return json.loads(r.read())["data"]

def control(dev, turn):
    api("POST", "/devices/control",
        {"device": dev["device"], "model": dev["model"], "command": "turn", "parameter": turn})
    print(f"{dev['deviceName']}: sent turn={turn}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for d in devices():
            print(f"{d['deviceName']:28s} {d['device']} controllable={d['controllable']}")
    elif cmd == "state":
        devs = devices()
        targets = [match(sys.argv[2])] if len(sys.argv) > 2 else devs
        for d in targets:
            props = {k: v for p in state_of(d)["properties"] for k, v in p.items()}
            print(f"{d['deviceName']:28s} power={props.get('powerState')} online={props.get('online')}")
    elif cmd in ("on", "off"):
        if len(sys.argv) < 3:
            sys.exit("usage: govee.py on|off <name>")
        control(match(sys.argv[2]), "on" if cmd == "on" else "off")
    else:
        sys.exit(__doc__)
