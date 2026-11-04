"""Token auth on the realtime server: the four ways a caller can present it.

Covers what a separate frontend needs to know:
  - WS with the token as a subprotocol (browsers cannot set headers)
  - WS with ?token= (curl, non-browser clients)
  - WS with a wrong or missing token -> 1008 without claiming a pipeline slot
  - /v1/pool gated, /v1/realtime/capabilities open

    python bench/probe_ws_auth.py

Reads the token from _ws_token, the same file run.ps1 generates.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import websockets

# S2S_URL points at the bare origin, e.g. wss://host, so the same probe can
# check the tunnel a frontend will actually use. The read routes are fetched
# over the same origin, with ws(s) swapped for http(s).
ORIGIN = os.environ.get("S2S_URL", "http://127.0.0.1:8765").rstrip("/")
URL = ORIGIN.replace("http", "ws", 1) + "/v1/realtime"
HTTP = ORIGIN.replace("ws", "http", 1)


def token() -> str:
    path = Path(__file__).resolve().parent.parent / "_ws_token"
    return path.read_text().strip() if path.is_file() else ""


async def connect(subprotocols=None, query=""):
    """Return (accepted, close_code, first_event_type).

    The server accepts the socket and then rejects a bad token, so "accepted"
    means the session was really established: the first event is session.created,
    not an error.
    """
    url = URL + (f"?{query}" if query else "")
    try:
        async with websockets.connect(url, subprotocols=subprotocols, max_size=None) as ws:
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            try:
                await asyncio.wait_for(ws.wait_closed(), timeout=2)
            except asyncio.TimeoutError:
                pass
            return first.get("type") != "error", None, first.get("type")
    except websockets.exceptions.InvalidStatus as exc:
        return False, exc.response.status_code, "handshake"
    except websockets.exceptions.ConnectionClosed as exc:
        return False, exc.rcvd.code if exc.rcvd else None, "closed"


async def main():
    tok = token()
    if not tok:
        print("no _ws_token file - start serve mode first")
        return 2
    results = []

    ok, _, first = await connect(["realtime", f"token.{tok}"])
    results.append(("subprotocol token", ok, True))
    print(f"subprotocol token      -> {'ACCEPTED' if ok else 'REJECTED'} (first event {first})")
    await asyncio.sleep(1.5)  # let the slot release

    ok, _, first = await connect(["realtime"], query=f"token={tok}")
    results.append(("query token", ok, True))
    print(f"query token            -> {'ACCEPTED' if ok else 'REJECTED'} (first event {first})")
    await asyncio.sleep(1.5)

    ok, code, first = await connect(["realtime", "token.wrong-token"])
    results.append(("wrong token", ok, False))
    print(f"wrong token            -> {'ACCEPTED' if ok else f'REJECTED ({code}, first event {first})'}")

    ok, code, first = await connect(["realtime"])
    results.append(("no token", ok, False))
    print(f"no token               -> {'ACCEPTED' if ok else f'REJECTED ({code}, first event {first})'}")

    import requests

    for path, want in (("/v1/pool", 401), ("/v1/usage", 401), ("/v1/realtime/capabilities", 200)):
        r = requests.get(f"{HTTP}{path}", timeout=10)
        results.append((f"GET {path}", r.status_code == want, True))
        print(f"GET {path:28s} -> {r.status_code} (want {want})")
    r = requests.get(f"{HTTP}/v1/pool", params={"token": tok}, timeout=10)
    results.append(("GET /v1/pool with token", r.status_code == 200, True))
    print(f"GET /v1/pool with token       -> {r.status_code}")

    # WebRTC is a second way into the same pipeline slots. Ungated it returns
    # 201 with an SDP answer to anyone who posts a valid offer; gated it must
    # answer 401 before the offer is even parsed.
    for method, path in (("POST", "/v1/realtime/calls"), ("DELETE", "/v1/realtime/calls/x")):
        r = requests.request(method, f"{HTTP}{path}", data="v=0",
                             headers={"Content-Type": "application/sdp"}, timeout=10)
        results.append((f"{method} {path}", r.status_code == 401, True))
        print(f"{method:6s} {path:24s} -> {r.status_code} (want 401)")

    bad = [name for name, got, want in results if got != want]
    if bad:
        print(f"\nFAIL: {', '.join(bad)}")
        return 1
    print("\nOK: token gates the socket and the read routes; capabilities stays open")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
