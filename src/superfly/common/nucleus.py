"""Nucleus (omniverse://) credentials for headless Isaac runs -- pure Python.

A headless Kit that opens an omniverse:// USD without credentials blocks on a
browser login forever (2026-09-29: PX4 waited for a heartbeat that never came;
2026-09-16: a stage composed empty and the drone fell through the void). The
AirLab Nucleus takes an API token as the password of the magic user
'$omni-api-token'; on airstation03 it lives in ~/.omni_env as
``export OMNI_API_TOKEN=<token>`` (chmod 600).

ensure_credentials() puts it into os.environ (OMNI_API_TOKEN, and
OMNI_USER / OMNI_PASS, which Kit's client library reads) BEFORE any Kit boots,
so every child sim inherits it; stat_usd() checks a URL with omni.client under
Isaac's python in a few seconds, so a broken token fails the campaign up front
instead of hanging its first trial.
"""
import os
import re
import subprocess
from pathlib import Path

OMNI_ENV = Path(os.environ.get("SUPERFLY_OMNI_ENV", str(Path.home() / ".omni_env")))
API_TOKEN_USER = "$omni-api-token"


def _token_from_file(path: Path):
    try:
        text = path.read_text()
    except OSError:
        return None
    m = re.search(r"^\s*(?:export\s+)?OMNI_API_TOKEN\s*=\s*['\"]?([^'\"\s]+)", text, re.M)
    return m.group(1) if m else None


def ensure_credentials() -> str:
    """Make the Nucleus credential visible to child processes. Returns where it
    came from; raises SystemExit (with the fix) if there is none."""
    if os.environ.get("OMNI_PASS") and os.environ.get("OMNI_USER"):
        if os.environ["OMNI_USER"] == API_TOKEN_USER:
            # run_px4_sim registers OMNI_API_TOKEN as its auth callback
            os.environ.setdefault("OMNI_API_TOKEN", os.environ["OMNI_PASS"])
        return "environment (OMNI_USER/OMNI_PASS)"
    tok = os.environ.get("OMNI_API_TOKEN") or _token_from_file(OMNI_ENV)
    if not tok:
        raise SystemExit(
            f"Nucleus credential missing: no OMNI_API_TOKEN / OMNI_PASS in the environment "
            f"and none in {OMNI_ENV}. Create an API token in the Nucleus web UI "
            "(https://airlab-nucleus.andrew.cmu.edu -> user menu -> API tokens) and write it "
            f"to {OMNI_ENV} as: export OMNI_API_TOKEN=<token>  (chmod 600).")
    src = "environment (OMNI_API_TOKEN)" if os.environ.get("OMNI_API_TOKEN") else str(OMNI_ENV)
    os.environ["OMNI_API_TOKEN"] = tok
    os.environ["OMNI_USER"] = API_TOKEN_USER
    os.environ["OMNI_PASS"] = tok
    return src


_STAT = r"""
import os, sys, glob
for d in glob.glob(os.path.join(sys.argv[2], "kit", "extscore", "omni.client.lib")):
    sys.path.insert(0, d)
import omni.client as oc
oc.initialize()
r, e = oc.stat(sys.argv[1])
print("NUCLEUS_STAT", r, getattr(e, "size", 0), flush=True)
sys.exit(0 if r == oc.Result.OK else 3)
"""


def stat_usd(url: str, sim_python: str, timeout: float = 180.0) -> str:
    """Stat `url` with omni.client under Isaac's python (`sim_python`, e.g.
    ~/isaacsim/python.sh). Returns the one-line result; raises SystemExit
    when the URL is unreachable or the credential is rejected."""
    if not url.startswith("omniverse://"):
        return "local"
    isaac_root = str(Path(sim_python).resolve().parent)
    try:
        out = subprocess.run([sim_python, "-c", _STAT, url, isaac_root], capture_output=True,
                             text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"Nucleus stat of {url} timed out after {timeout:.0f} s")
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("NUCLEUS_STAT")), "")
    if out.returncode != 0 or not line:
        raise SystemExit(f"Nucleus stat of {url} failed ({line or out.stderr[-400:]}): the "
                         f"token in {OMNI_ENV} no longer authenticates, or the server is down.")
    return line
