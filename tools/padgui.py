#!/usr/bin/env python3
"""Tiny web gamepad for driving the virtual pad by hand - menu navigation over
VNC is slow, this is instant.

    python3 tools/padgui.py            # http://127.0.0.1:8090
    python3 tools/padgui.py 8091       # other port

Every button opens a one-shot TCP connection to the pad server and sends one
line of its text protocol (see control/virtual_pad_server.py):

    nav up|down|left|right [ms]     d-pad / stick tap
    press a|b|x|y|lb|rb|start|select [ms]
    steer <-1..1> / gas <0..1> / brake <0..1>
    reset

Seat selector picks which pad (8765 / 8775 / 8785 / 8795). For splitscreen
menu setup you only ever need seat 0 - player 1 drives every menu.
"""
import re
import shutil
import socket
import subprocess
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
PAD_HOST = "127.0.0.1"
# The dev game runs on this X display; xdotool works on a real Xvfb (it does
# NOT on Wayland :0). Override with: python3 tools/padgui.py 8090 :100
GAME_DISPLAY = sys.argv[2] if len(sys.argv) > 2 else ":99"
XDOTOOL = shutil.which("xdotool")


def _xdo(args, disp=None):
    """Run xdotool against a game display, focusing the TM window first so
    keystrokes land in the on-screen text field (unlike the pad, real key
    events need focus)."""
    if not XDOTOOL:
        return "xdotool not installed"
    if disp and not re.fullmatch(r":\d+(\.\d+)?", disp):
        return f"bad display {disp!r}"
    env = {"DISPLAY": disp or GAME_DISPLAY, "PATH": "/usr/bin:/bin"}
    try:
        wid = subprocess.run(
            [XDOTOOL, "search", "--name", "Trackmania"],
            env=env, capture_output=True, text=True, timeout=3
        ).stdout.split()
        if wid:
            subprocess.run([XDOTOOL, "windowactivate", "--sync", wid[0]],
                           env=env, capture_output=True, timeout=3)
        subprocess.run([XDOTOOL] + args, env=env, capture_output=True,
                       timeout=5, check=True)
        return "ok"
    except subprocess.CalledProcessError as e:
        return f"xdotool err: {e.stderr.decode(errors='ignore')[:120] if e.stderr else e}"
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"xdotool err: {e}"

PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<title>pad</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#111;color:#eee;font:15px/1.3 system-ui,sans-serif;
      -webkit-user-select:none;user-select:none;touch-action:manipulation}
 header{display:flex;gap:12px;align-items:center;padding:10px 12px;background:#1b1b1b;
        position:sticky;top:0;flex-wrap:wrap}
 select,input{background:#222;color:#eee;border:1px solid #444;border-radius:6px;
              padding:6px 8px;font:inherit}
 #log{font:12px/1.4 ui-monospace,monospace;color:#8a8;padding:4px 12px;min-height:18px}
 .wrap{display:flex;gap:26px;flex-wrap:wrap;padding:16px;justify-content:center}
 .pad{display:grid;gap:8px}
 .dpad{grid-template-columns:repeat(3,72px);grid-template-rows:repeat(3,72px)}
 .face{grid-template-columns:repeat(3,72px);grid-template-rows:repeat(3,72px)}
 button{background:#2a2a2a;color:#eee;border:1px solid #555;border-radius:12px;
        font:600 18px system-ui;cursor:pointer}
 button:active{background:#3d6;color:#111;transform:scale(.96)}
 .dpad button,.face button{width:72px;height:72px}
 .mid{display:flex;gap:10px;align-items:center;justify-content:center;flex-wrap:wrap}
 .mid button{padding:12px 16px;font-size:15px}
 .drive{display:flex;gap:10px;padding:0 16px 24px;justify-content:center;flex-wrap:wrap}
 .drive button{padding:16px 20px;font-size:15px;border-radius:12px}
 .kb{display:flex;gap:8px;align-items:center;justify-content:center;flex-wrap:wrap;
     padding:0 16px 18px}
 .kb button{padding:12px 14px;font-size:14px;border-radius:10px}
 .kb input{width:180px;font-size:16px;text-align:center}
 .hint{color:#888;font-size:12px;text-align:center;padding:0 16px 20px}
 kbd{background:#333;border:1px solid #555;border-radius:4px;padding:1px 5px;font:12px monospace}
</style></head><body>
<header>
 <label>instance <select id=inst>
   <option value=0>0 &mdash; dev :99</option>
   <option value=1>1 &mdash; school :100</option>
   <option value=2>2 &mdash; :101</option>
 </select></label>
 <label>seat <select id=seat>
   <option value=0>0</option><option value=1>1</option>
   <option value=2>2</option><option value=3>3</option>
 </select></label>
 <label title="leave blank to use instance+seat; set to force a specific pad port">port <input id=portovr type=number placeholder=auto style=width:78px></label>
 <span class=hint id=portlbl style="margin:0">pad 8765 &middot; kb :99</span>
 <label>nav ms <input id=ms type=number value=90 style=width:64px></label>
 <button id=reset style="padding:8px 12px;font-size:14px">reset pad</button>
</header>
<div id=log>ready</div>
<div class=wrap>
 <div class="pad dpad">
  <span></span><button data-nav=up>▲</button><span></span>
  <button data-nav=left>◄</button><button data-nav=down>▼</button><button data-nav=right>►</button>
  <span></span><span></span><span></span>
 </div>
 <div class="pad face">
  <span></span><button data-btn=y>Y</button><span></span>
  <button data-btn=x>X</button><span></span><button data-btn=b>B</button>
  <span></span><button data-btn=a>A</button><span></span>
 </div>
</div>
<div class=mid>
 <button data-btn=lb>LB</button>
 <button data-btn=select>SELECT / ⧉</button>
 <button data-btn=start>START / ☰ (options)</button>
 <button data-btn=rb>RB</button>
</div>
<div class=drive>
 <button data-hold="steer -0.8">◄ steer</button>
 <button data-hold="steer 0.8">steer ►</button>
 <button data-hold="gas 1">GAS</button>
 <button data-hold="brake 1">BRAKE</button>
</div>
<div class=kb>
 <button data-key="ctrl+a">Ctrl+A</button>
 <button data-key=BackSpace>⌫</button>
 <input id=txt inputmode=numeric placeholder="time limit e.g. 999999" autocomplete=off>
 <button id=typebtn>type</button>
 <button data-key=Return>Enter ⏎</button>
 <span style="color:#888;font-size:12px">keyboard follows the instance selector above</span>
</div>
<div class=hint>
 keys: <kbd>← ↑ → ↓</kbd> nav &nbsp; <kbd>Enter</kbd> A &nbsp; <kbd>Backspace</kbd> B
 &nbsp; <kbd>x</kbd> <kbd>y</kbd> &nbsp; <kbd>[</kbd> LB <kbd>]</kbd> RB
 &nbsp; <kbd>Space</kbd> START &nbsp; <kbd>\</kbd> SELECT
</div>
<script>
const $=s=>document.querySelector(s), log=$('#log');
// pad port + keyboard display, from instance + seat:
//   instance 0 seats -> 8765 + 10*seat   (splitscreen game on :99)
//   instance N>0 seat -> 8900 + 10*((N-1)*4 + seat)   (matches env/ports.seat_ports)
//   keyboard display   -> :99 for game 0, :(99+N) otherwise
function inst(){return +$('#inst').value;}
function port(){
  const ovr=$('#portovr').value.trim();
  if(ovr) return +ovr;
  const n=inst(), s=+$('#seat').value;
  return n===0 ? 8765+10*s : 8900+10*((n-1)*4+s);
}
function disp(){const n=inst(); return n===0 ? ':99' : ':'+(99+n);}
function syncLbl(){$('#portlbl').textContent='pad '+port()+' · kb '+disp();}
const ms=()=>$('#ms').value||90;
addEventListener('DOMContentLoaded',()=>{
  $('#inst').onchange=syncLbl; $('#seat').onchange=syncLbl;
  $('#portovr').oninput=syncLbl; syncLbl();
});
async function send(cmd){
  try{const r=await fetch('/send?port='+port()+'&cmd='+encodeURIComponent(cmd));
      log.textContent=cmd+'  ->  '+(await r.text()).trim();}
  catch(e){log.textContent=cmd+'  ERR '+e;}
}
document.querySelectorAll('[data-nav]').forEach(b=>
  b.onclick=()=>send('nav '+b.dataset.nav+' '+ms()));
document.querySelectorAll('[data-btn]').forEach(b=>
  b.onclick=()=>send('press '+b.dataset.btn));
$('#reset').onclick=()=>send('reset');
// hold-to-apply for the drive row (mouse + touch)
document.querySelectorAll('[data-hold]').forEach(b=>{
  const on=e=>{e.preventDefault();send(b.dataset.hold);};
  const off=()=>{const a=b.dataset.hold.split(' ')[0];send(a+' 0');};
  b.addEventListener('mousedown',on); b.addEventListener('touchstart',on,{passive:false});
  b.addEventListener('mouseup',off); b.addEventListener('mouseleave',off);
  b.addEventListener('touchend',off);
});
// --- keyboard (xdotool -> the selected instance's display) ---
async function kb(path){
  const sep = path.includes('?') ? '&' : '?';
  try{const r=await fetch(path+sep+'disp='+encodeURIComponent(disp()));
      log.textContent=path+'  ->  '+(await r.text()).trim();}
  catch(e){log.textContent=path+'  ERR '+e;}
}
document.querySelectorAll('[data-key]').forEach(b=>
  b.onclick=()=>kb('/key?keys='+encodeURIComponent(b.dataset.key)));
$('#typebtn').onclick=()=>{
  const t=$('#txt').value; if(t!=='') kb('/type?text='+encodeURIComponent(t));
};
$('#txt').addEventListener('keydown',e=>{
  if(e.key==='Enter'){e.preventDefault();$('#typebtn').click();}
});
const KEYS={ArrowUp:'nav up',ArrowDown:'nav down',ArrowLeft:'nav left',ArrowRight:'nav right',
 Enter:'press a',Backspace:'press b',' ':'press start','\\':'press select',
 x:'press x',y:'press y','[':'press lb',']':'press rb'};
addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  let c=KEYS[e.key]; if(!c)return; e.preventDefault();
  if(c.startsWith('nav'))c+=' '+ms(); send(c);
});
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/plain"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/key":
            q = urllib.parse.parse_qs(u.query)
            keys = q.get("keys", [""])[0].strip()
            disp = q.get("disp", [None])[0]
            if not keys:
                return self._send(400, "empty keys")
            return self._send(200, _xdo(["key", "--clearmodifiers", keys], disp))
        if u.path == "/type":
            q = urllib.parse.parse_qs(u.query)
            text = q.get("text", [""])[0]
            disp = q.get("disp", [None])[0]
            if text == "":
                return self._send(400, "empty text")
            return self._send(200, _xdo(
                ["type", "--clearmodifiers", "--delay", "30", text], disp))
        if u.path == "/send":
            q = urllib.parse.parse_qs(u.query)
            try:
                port = int(q.get("port", ["8765"])[0])
                cmd = q.get("cmd", [""])[0].strip()
                if not cmd:
                    return self._send(400, "empty cmd")
                with socket.create_connection((PAD_HOST, port), timeout=2) as s:
                    s.sendall((cmd + "\n").encode())
                    s.settimeout(1.0)
                    try:
                        reply = s.recv(256).decode(errors="ignore").strip()
                    except socket.timeout:
                        reply = "(no reply)"
                return self._send(200, reply or "ok")
            except OSError as e:
                return self._send(502, f"pad {q.get('port')} unreachable: {e}")
        return self._send(404, "not found")


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"pad gui on http://127.0.0.1:{PORT}  (pads {PAD_HOST}:8765/8775/8785/8795)",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
