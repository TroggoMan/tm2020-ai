# Instance grid — live video of every game, in one resizable page

`http://127.0.0.1:8080/grid` (also the **Instance grid** button in the panel).

One MJPEG feed per X display, tiled in a browser grid you can rearrange. No
noVNC, no websockify — ffmpeg's `mpjpeg` muxer emits exactly the
`multipart/x-mixed-replace` an `<img>` renders on its own, so the panel just
grabs each display and pipes it through.

## Using it

- **cols / fps / width** in the header set the layout and the per-feed encode.
  Lower fps + width if the CPU notices; 12fps / 960px is fine for monitoring.
- **Drag a pane's title bar** to reorder. **Drag a pane's bottom-right corner**
  to resize it. Layout, sizes, hidden panes and the header settings are saved
  in the browser (localStorage) and restored on reload.
- **✕** on a pane hides it; press **r** (not in a field) to bring hidden panes
  back.
- **clean view** hides the header and pane chrome — the page becomes a clean
  capture surface. **fullscreen** does the obvious thing.
- It rescans for displays every 30s, so instances that start or stop appear
  and disappear on their own.

Displays come from `/tmp/.X11-unix`. `:99` is the headless main game
(`tools/headless-main.sh up`); fleet instances are `:100+`
(`tools/steam-instance <n> --vnc`). A display the panel's user has no xauth
for shows as "(no window)" but still streams if ffmpeg can grab it.

## Streaming the grid as one screen

The grid page *is* the composite — arrange it once, then capture that single
surface instead of `tools/stream.py`'s fixed xstack:

```
# 1. a dedicated display for the grid browser
Xvfb :90 -screen 0 1920x1080x24 &
DISPLAY=:90 openbox &
DISPLAY=:90 chromium --kiosk --app=http://127.0.0.1:8080/grid &
#    (in the page: arrange panes, click "clean view")

# 2. stream that one display - TAS stamp + nvenc as always
tools/stream.py --single-display :90 --twitch <key>
tools/stream.py --single-display :90 --preview        # local file first
```

`tools/stream.py` with no `--single-display` still does the old thing: grab
`:99, :100, …` directly and xstack them into a fixed grid. Use that when you
want the tidy 2x2 and don't care about rearranging; use `--single-display`
when you want the layout you dragged in the browser.
