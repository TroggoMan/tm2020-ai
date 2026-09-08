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
import os
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
# Keysyms are passed to xdotool, so constrain them rather than trusting the
# browser: letters/digits, and the named keys X uses (F3, Escape, Return...).
KEYSYM_OK = re.compile(r"^[A-Za-z0-9_+]{1,24}$")


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
<div class=row style="align-items:center;gap:10px;flex-wrap:wrap">
 <button id=ptbtn style="font-weight:700">Passthrough: OFF</button>
 <label style="font-size:13px">auto-off
  <select id=ptsecs>
   <option value=30>30s</option><option value=60 selected>60s</option>
   <option value=180>3m</option><option value=600>10m</option>
  </select></label>
 <span id=ptleft style="color:#888;font-size:12px"></span>
 <span style="color:#888;font-size:12px">Esc exits. Every key goes to the game, including F3.</span>
</div>
<div class=row style="gap:10px;flex-wrap:wrap">
 <div id=trackpad style="width:260px;height:150px;background:#191919;border:1px solid #333;
      border-radius:8px;display:flex;align-items:center;justify-content:center;
      color:#666;font-size:12px;touch-action:none">drag = move mouse</div>
 <div style="display:flex;flex-direction:column;gap:6px">
  <button data-mb=1>left click</button>
  <button data-mb=3>right click</button>
  <button data-key=F3 style="font-weight:700">F3 (Openplanet)</button>
 </div>
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
  if(passthrough)return;               // passthrough owns the keyboard
  if(e.target.tagName==='INPUT')return;
  let c=KEYS[e.key]; if(!c)return; e.preventDefault();
  if(c.startsWith('nav'))c+=' '+ms(); send(c);
});

// --- raw keyboard passthrough ------------------------------------------
//
// Everything you press goes to the game's X display as a real key event, held
// keys included - which is the only way to reach F3 and drive the Openplanet
// overlay when you are not sitting at the machine. It grabs the whole
// keyboard, so it ALWAYS has a way out: Escape, the button, or the auto-off
// timer. Getting stuck with no keyboard on a remote box is worse than not
// having the feature.
let passthrough=false, ptTimer=null, ptEnd=0, ptHeld=new Set();
const NAMED={Escape:'Escape',Enter:'Return',Backspace:'BackSpace',Tab:'Tab',
 ' ':'space',ArrowUp:'Up',ArrowDown:'Down',ArrowLeft:'Left',ArrowRight:'Right',
 Delete:'Delete',Home:'Home',End:'End',PageUp:'Prior',PageDown:'Next',
 Shift:'Shift_L',Control:'Control_L',Alt:'Alt_L',Meta:'Super_L'};
function keysym(e){
  if(/^F\d{1,2}$/.test(e.key))return e.key;
  if(NAMED[e.key])return NAMED[e.key];
  if(e.key.length===1){
    const c=e.key;
    if(/[A-Za-z0-9]/.test(c))return c;
    const P={'-':'minus','=':'equal','[':'bracketleft',']':'bracketright',
      ';':'semicolon',"'":'apostrophe',',':'comma','.':'period','/':'slash',
      '\\':'backslash','`':'grave'};
    return P[c]||null;
  }
  return null;
}
async function ptSend(path,k){
  try{await fetch(path+'?k='+encodeURIComponent(k)+'&disp='+encodeURIComponent(disp()));}
  catch(e){}
}
function ptTick(){
  const left=Math.max(0,Math.ceil((ptEnd-Date.now())/1000));
  $('#ptleft').textContent=passthrough?('auto-off in '+left+'s'):'';
  if(passthrough&&left<=0)ptSet(false);
}
function ptSet(on){
  passthrough=on;
  $('#ptbtn').textContent='Passthrough: '+(on?'ON':'OFF');
  $('#ptbtn').style.background=on?'#4d3':'';
  $('#ptbtn').style.color=on?'#111':'';
  if(on){ptEnd=Date.now()+(+$('#ptsecs').value)*1000;
    if(!ptTimer)ptTimer=setInterval(ptTick,250);}
  else{ // never leave a key stuck down on the game
    ptHeld.forEach(k=>ptSend('/keyup',k)); ptHeld.clear();
    clearInterval(ptTimer); ptTimer=null; $('#ptleft').textContent='';}
  ptTick();
}
$('#ptbtn').onclick=()=>ptSet(!passthrough);
addEventListener('keydown',e=>{
  if(!passthrough)return;
  if(e.key==='Escape'){e.preventDefault();ptSet(false);return;}
  const k=keysym(e); if(!k)return;
  e.preventDefault();
  ptEnd=Date.now()+(+$('#ptsecs').value)*1000;   // activity extends the timer
  if(!ptHeld.has(k)){ptHeld.add(k);ptSend('/keydown',k);}
});
addEventListener('keyup',e=>{
  if(!passthrough)return;
  const k=keysym(e); if(!k)return;
  e.preventDefault();
  if(ptHeld.delete(k))ptSend('/keyup',k);
});
addEventListener('blur',()=>{if(passthrough)ptSet(false);});

// --- mouse -------------------------------------------------------------
(function(){
  const tp=$('#trackpad'); let last=null;
  const move=(x,y)=>{
    if(last){const dx=x-last[0],dy=y-last[1];
      if(Math.abs(dx)>0||Math.abs(dy)>0)
        fetch('/mousemove?dx='+Math.round(dx)+'&dy='+Math.round(dy)
              +'&disp='+encodeURIComponent(disp())).catch(()=>{});}
    last=[x,y];
  };
  tp.addEventListener('pointerdown',e=>{tp.setPointerCapture(e.pointerId);last=[e.clientX,e.clientY];});
  tp.addEventListener('pointermove',e=>{if(last)move(e.clientX,e.clientY);});
  tp.addEventListener('pointerup',()=>{last=null;});
  document.querySelectorAll('[data-mb]').forEach(b=>b.onclick=()=>
    fetch('/click?b='+b.dataset.mb+'&disp='+encodeURIComponent(disp())).catch(()=>{}));
})();
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
        if u.path in ("/keydown", "/keyup"):
            # Held keys need down/up as separate events - `xdotool key` taps,
            # which is wrong for anything you hold (menus that scroll, a key
            # the game samples per frame).
            q = urllib.parse.parse_qs(u.query)
            k = q.get("k", [""])[0].strip()
            disp = q.get("disp", [None])[0]
            if not k or len(k) > 24 or not KEYSYM_OK.match(k):
                return self._send(400, "bad keysym")
            verb = "keydown" if u.path == "/keydown" else "keyup"
            return self._send(200, _xdo([verb, k], disp))
        if u.path == "/mousemove":
            q = urllib.parse.parse_qs(u.query)
            disp = q.get("disp", [None])[0]
            try:
                dx = max(-400, min(400, int(float(q.get("dx", ["0"])[0]))))
                dy = max(-400, min(400, int(float(q.get("dy", ["0"])[0]))))
            except ValueError:
                return self._send(400, "bad delta")
            return self._send(200, _xdo(
                ["mousemove_relative", "--", str(dx), str(dy)], disp))
        if u.path == "/click":
            q = urllib.parse.parse_qs(u.query)
            disp = q.get("disp", [None])[0]
            b = q.get("b", ["1"])[0]
            if b not in ("1", "2", "3", "4", "5"):
                return self._send(400, "bad button")
            return self._send(200, _xdo(["click", b], disp))
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
    BIND = os.environ.get("PADGUI_BIND", "0.0.0.0")
    if BIND not in ("127.0.0.1", "localhost"):
        print(f"  !! binding {BIND}:{PORT} - this drives the game's keyboard, "
              f"mouse and gamepad. No auth. LAN/VPN only.", flush=True)
    srv = ThreadingHTTPServer((BIND, PORT), H)
    print(f"pad gui on http://127.0.0.1:{PORT}  (pads {PAD_HOST}:8765/8775/8785/8795)",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
