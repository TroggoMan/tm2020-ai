"""Where each instance's Wine prefix lives.

One module because three tools need the same answer: the prefix cloner, the
map pusher, and anything that later wants to read a per-instance config.

Instance 0 is the existing Steam prefix, untouched - the setup you already
have keeps working exactly as it did. Instances 1+ are clones of it, placed
next to it on the same filesystem rather than under $HOME, for one concrete
reason: /mnt/4TB is XFS with reflink, so `cp --reflink` clones 2.3GB in
about a second and costs no extra disk until the copies diverge. Putting them
on /home would cross a filesystem boundary and turn each clone into a real
2.3GB write.
"""
from __future__ import annotations

import os

STEAM_APPID = "2225070"
STEAM_LIBRARY = "/mnt/4TB/SteamLibrary"
BASE_COMPAT = os.path.join(STEAM_LIBRARY, "steamapps", "compatdata", STEAM_APPID)
GAME_DIR = os.path.join(STEAM_LIBRARY, "steamapps", "common", "Trackmania")
CLONE_ROOT = os.environ.get(
    "TMAI_PREFIX_ROOT", "/mnt/4TB/tm2020-ai-prefixes")

# Inside a prefix.
DOCS = "drive_c/users/steamuser/Documents/Trackmania"
OPENPLANET = "drive_c/users/steamuser/OpenplanetNext"
UBI_SESSION = ("drive_c/users/steamuser/AppData/Local/"
               "Ubisoft Game Launcher/user.dat")


def compat_dir(i: int) -> str:
    """The compatdata directory - the thing Proton is pointed at.

    Note this is the *parent* of `pfx`: Proton wants STEAM_COMPAT_DATA_PATH to
    be the directory containing pfx, version and config_info, not pfx itself.
    """
    if i == 0:
        return os.environ.get("TMAI_PREFIX_0", BASE_COMPAT)
    return os.path.join(CLONE_ROOT, f"instance-{i:02d}")


def prefix_for(i: int) -> str:
    return os.path.join(compat_dir(i), "pfx")


def docs_dir(i: int) -> str:
    return os.path.join(prefix_for(i), DOCS)


def exists(i: int) -> bool:
    return os.path.isdir(docs_dir(i))
