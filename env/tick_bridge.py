"""Bridge to the TICK TAS tool: pre-scheduled inputs, state clips, fast-forward.

See TICK.md for the protocol notes this is built on. The single most important
thing this module exists to fix: `tools/headless-main.sh map` returns BEFORE the
map has reloaded, so naive code straddles a restart and measures the stale run.
`restart()` below blocks until the new run is genuinely live.
"""
import json, re, socket, struct, sqlite3, subprocess, time, urllib.request, urllib.error

TICK_DIR = "/home/troggoman/.local/share/tick"
DB = "file:/home/troggoman/.local/share/tick-tas-tool/data/tas-data.db?mode=ro"
BASE = "http://127.0.0.1:46321/api"
BROKER = ("127.0.0.1", 8767)
STRIDE = 2434          # frame stride, verified by correlation (TICK.md)
POS_OFF, VEL_OFF, QUAT_OFF = 84, 96, 68


def _key():
    # regenerated every launcher start - never hardcode
    src = open(f"{TICK_DIR}/ui/runtime-config.js").read()
    return re.search(r'"dataServiceApiKey":"([0-9A-F]+)"', src).group(1)


def req(path, method="GET", body=None, raw=None, ctype="application/json"):
    data = raw.encode() if raw is not None else (json.dumps(body).encode() if body is not None else None)
    r = urllib.request.Request(BASE + path, method=method, data=data,
                               headers={"X-Tick-Data-Service-Key": _key(), "Content-Type": ctype})
    try:
        d = urllib.request.urlopen(r, timeout=15).read()
        try:
            return json.loads(d) if d else None
        except ValueError:
            return d.decode("utf8", "replace")
    except urllib.error.HTTPError as e:
        return {"__error": e.code, "body": e.read().decode("utf8", "replace")[:300]}


def telemetry():
    """One frame from the TMAITelemetry broker. May lack race_time between runs."""
    s = socket.create_connection(BROKER, 5)
    s.settimeout(8)
    buf = b""
    while b"\n" not in buf:
        buf += s.recv(65536)
    s.close()
    return json.loads(buf.split(b"\n")[0])


def live():
    f = telemetry()
    return f if (f.get("car") and f.get("in_race") and f.get("race_time") is not None) else None


def wait_cmd(cid, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for c in req("/runtime/commands?limit=10&newestFirst=true") or []:
            if c["id"] == cid and c["status"] in ("succeeded", "failed"):
                return c
        time.sleep(0.05)
    raise TimeoutError(f"command {cid} did not settle")


def set_speed(n):
    req("/runtime/game-speed", "PUT", {"requestedGameSpeed": n})
    for _ in range(40):
        time.sleep(0.1)
        if req("/runtime/status")["effectiveGameSpeed"] == n:
            return n
    raise RuntimeError(f"game speed {n} not applied")


def patch_settings(**kw):
    st = req("/settings")
    kw["expectedRevision"] = st["revision"]
    return req("/settings", "PATCH", kw)


def make_revision(collection_id, script, retries=3):
    """Create a revision from TAS script text. Retries: the endpoint returns a
    transient error occasionally, and blindly indexing ["revision"] turns that
    into a confusing KeyError far from the cause."""
    last = None
    for _ in range(retries):
        r = req(f"/input-collections/{collection_id}/revisions?origin=bridge",
                "POST", raw=script, ctype="text/plain; charset=utf-8")
        if isinstance(r, dict) and "revision" in r:
            return r["revision"]["id"]
        last = r
        time.sleep(0.6)
    raise RuntimeError(f"make_revision failed after {retries} tries: {last}")


def load_revision(collection_id, rev_id):
    c = req(f"/input-collections/{collection_id}")
    crev = c.get("node", c)["revision"]          # 400s without this
    return req(f"/input-revisions/{rev_id}/load?expectedCollectionRevision={crev}", "POST")


def restart(map_path, repo="/home/troggoman/tm2020-ai", timeout=120):
    """Reload the map and BLOCK until the new run is live at tick 0.

    Playback only arms at tick 0, so every scripted run must go through here.
    """
    before = (live() or {}).get("race_time")
    subprocess.run(["tools/headless-main.sh", "map", map_path],
                   cwd=repo, capture_output=True, timeout=timeout)
    t0 = time.time()
    saw_gap = before is None
    while time.time() - t0 < timeout:
        f = live()
        if f is None:
            saw_gap = True                       # car vanished => reload in progress
        elif saw_gap and f["race_time"] < (before or 10**9):
            return f                             # fresh run, genuinely at the start
        time.sleep(0.01)
    raise TimeoutError("map restart never produced a fresh run")


def run_until(race_time, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        f = live()
        if f and f["race_time"] >= race_time:
            return f
        time.sleep(0.005)
    raise TimeoutError(f"race_time {race_time} not reached")


def capture_clip(map_uid):
    c = req("/runtime/capture-state-clip?requestedBy=bridge", "POST", {"expectedMapUid": map_uid})
    return wait_cmd(c["id"])["result"]["runtimeClipId"]


def persist_clip(runtime_clip_id, map_uid):
    p = req("/runtime/persist-state-clip?requestedBy=bridge", "POST",
            {"runtimeClipId": runtime_clip_id, "expectedMapUid": map_uid})
    return wait_cmd(p["id"])["result"]["stateClipId"]


def replay_clip(runtime_clip_id, map_uid):
    r = req("/runtime/replay-state-clip?requestedBy=bridge", "POST",
            {"runtimeClipId": runtime_clip_id, "expectedMapUid": map_uid})
    return wait_cmd(r["id"])


def clip_blob(state_clip_id):
    db = sqlite3.connect(DB, uri=True)
    return b"".join(x[0] for x in db.execute(
        "SELECT bytes FROM state_clip_payload_chunks WHERE state_clip_id=? ORDER BY chunk_index",
        (state_clip_id,)))


def trajectory(blob, ground_truth_pos, frames=100):
    """Decode a clip into per-frame (pos, vel).

    The header length VARIES between clips, so it is recovered by correlation
    against a telemetry position sampled at capture time - never hardcoded.
    Returns (frames_list, header, gt_frame_index).
    """
    px, py, pz = ground_truth_pos
    hits = [o for o in range(len(blob) - 12)
            if abs(struct.unpack_from("<f", blob, o)[0] - px) < 0.02
            and abs(struct.unpack_from("<f", blob, o + 4)[0] - py) < 0.02
            and abs(struct.unpack_from("<f", blob, o + 8)[0] - pz) < 0.02]
    if not hits:
        raise ValueError("ground-truth position not found in clip; cannot locate header")
    H = (hits[0] - POS_OFF) % STRIDE
    gt_frame = (hits[0] - POS_OFF - H) // STRIDE
    out = []
    for i in range(frames):
        b = H + i * STRIDE
        out.append({"pos": struct.unpack_from("<3f", blob, b + POS_OFF),
                    "vel": struct.unpack_from("<3f", blob, b + VEL_OFF),
                    "quat": struct.unpack_from("<4f", blob, b + QUAT_OFF)})
    return out, H, gt_frame


def script_from_actions(actions, seed=None):
    """actions: [(time_ms, 'accel'|'brake'|'steer', value)]. steer is +/-127."""
    lines = [f"0 seed {seed}"] if seed is not None else []
    lines += [f"{int(t)} {k} {int(v)}" for t, k, v in actions]
    return "\n".join(lines) + "\n"
