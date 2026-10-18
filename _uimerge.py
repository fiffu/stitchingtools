"""
_merge_ui.py: single-file, offline web UI for compositing layered PNGs.

Usage:
    python _merge_ui.py [--host 127.0.0.1] [--port 0]

Put this file next to your PNGs (or in a parent of the folders that hold them),
run it, and open the printed URL.

Requires: pillow. Optional: numpy + blend_modes (exact multiply blending, same
as _merge.py; without them a Pillow-only approximation is used).

Files are grouped into layers by filename: the LAST run of digits plus anything
trailing it is the variant, everything before it is the layer name.
    B1_i000_2p -> layer 'B1_i000_', variant '2p'
Layers are stacked in reverse-alphabetical order in the picker (top = first),
which is the order _merge.py expects in its input file.

The outbox is '_info.txt' in the loaded directory, in _merge.py's format.
"""

import argparse
import hashlib
import io
import json
import os
import re
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageChops

try:
    import numpy
    from blend_modes import multiply as _bm_multiply
    HAVE_BLEND_MODES = True
except ImportError:
    HAVE_BLEND_MODES = False

ROOT = os.path.dirname(os.path.realpath(__file__))
EXT = '.png'
INFO_NAME = '_info.txt'
CACHE_DIR = os.path.join(ROOT, '.merge_ui_cache')

COMMENT_PREFIX = '#'
MULTIPLY_SUFFIX = '*'
PASTE_POSITION = '@'
PASTE_POSITION_DELIM = ','


class Err(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


# ---------------------------------------------------------------- filesystem

def resolve_dir(p):
    """Resolve a user-supplied dir; must be ROOT or a subdirectory of it."""
    full = os.path.realpath(os.path.join(ROOT, p or '.'))
    if full != ROOT and not full.startswith(ROOT + os.sep):
        raise Err(403, 'Directory must be inside ' + ROOT)
    if not os.path.isdir(full):
        raise Err(404, 'Not a directory: ' + full)
    return full


def safe_file(d, stem):
    if not stem or stem != os.path.basename(stem) or '\\' in stem:
        raise Err(400, 'Bad file name: ' + stem)
    path = os.path.join(d, stem + EXT)
    if not os.path.isfile(path):
        raise Err(404, 'Missing file: ' + stem + EXT)
    return path


# ------------------------------------------------------------------ grouping

# Edit this to change how files are grouped into layers/variants.
# Returns (layer_name, variant_label, sort_key).
_SPLIT = re.compile(r'^(.*?)(\d+)(\D*)$')


def split_stem(stem):
    m = _SPLIT.match(stem)
    if not m:
        return stem, stem, (0, 0, '')
    pre, digits, tail = m.groups()
    return pre, digits + tail, (int(digits), len(digits), tail)


def natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def list_layers(d):
    groups = {}
    for fn in os.listdir(d):
        if fn.startswith('.') or not fn.endswith(EXT):
            continue
        stem = fn[:-len(EXT)]
        name, label, key = split_stem(stem)
        groups.setdefault(name, []).append((key, stem, label))
    out = []
    for name in sorted(groups, key=natkey, reverse=True):
        variants = sorted(groups[name])
        out.append({'name': name, 'variants': [[s, l] for _, s, l in variants]})
    return out


def list_dirs():
    """Directories under ROOT (depth <= 3) that contain PNGs, for autocomplete."""
    found = []
    for cur, subs, files in os.walk(ROOT):
        subs[:] = sorted(s for s in subs if not s.startswith('.'))
        depth = 0 if cur == ROOT else os.path.relpath(cur, ROOT).count(os.sep) + 1
        if depth >= 3:
            subs[:] = []
        if cur == ROOT or any(f.endswith(EXT) for f in files):
            found.append(cur)
    return found


# ----------------------------------------------------------------- compositing

class LayerSpec:
    def __init__(self, file, multiply, offset):
        self.file, self.multiply, self.offset = file, multiply, offset


def parse_line(line):
    """Same notation as _merge.py: name[*][@x,y] # comment"""
    line = line.split(COMMENT_PREFIX, 1)[0].strip()
    multiply = MULTIPLY_SUFFIX in line
    line = line.replace(MULTIPLY_SUFFIX, '')
    file, *offset = line.split(PASTE_POSITION, 1)
    file = file.strip()
    if offset:
        try:
            offset = tuple(map(int, offset[0].split(PASTE_POSITION_DELIM, 1)))
        except ValueError:
            raise Err(400, 'Bad offset in line: ' + line)
    else:
        offset = ()
    return LayerSpec(file, multiply, offset)


def blend_multiply(base, new):
    if HAVE_BLEND_MODES:
        b = numpy.array(base).astype(float)
        n = numpy.array(new).astype(float)
        return Image.fromarray(numpy.uint8(_bm_multiply(b, n, 1)))
    mult = ImageChops.multiply(base.convert('RGB'), new.convert('RGB'))
    mixed = Image.composite(mult, base.convert('RGB'), new.getchannel('A')).convert('RGBA')
    mixed.putalpha(base.getchannel('A'))
    return mixed


def compose(d, lines, maxw=None):
    specs = [s for s in map(parse_line, lines) if s.file]
    if not specs:
        raise Err(400, 'No layers')
    bottom_first = specs[::-1]
    first = bottom_first.pop(0)
    base = Image.open(safe_file(d, first.file)).convert('RGBA')
    for spec in bottom_first:
        layer = Image.open(safe_file(d, spec.file)).convert('RGBA')
        if spec.multiply and spec.offset:
            continue  # unsupported combination, skipped like _merge.py
        if spec.offset:
            tmp = Image.new('RGBA', base.size)
            tmp.paste(layer, spec.offset)
            base.alpha_composite(tmp)
        elif spec.multiply:
            base = blend_multiply(base, layer)
        else:
            base.alpha_composite(layer)
    if maxw and base.width > maxw:
        base = base.resize((maxw, round(base.height * maxw / base.width)), Image.LANCZOS)
    buf = io.BytesIO()
    base.save(buf, 'PNG')
    return buf.getvalue()


_preview_cache = OrderedDict()
_preview_lock = threading.Lock()


def preview_bytes(d, lines, maxw):
    stems = [s.file for s in map(parse_line, lines) if s.file]
    mt = tuple(os.stat(safe_file(d, s)).st_mtime_ns for s in stems)
    key = (d, tuple(lines), maxw, mt)
    with _preview_lock:
        if key in _preview_cache:
            _preview_cache.move_to_end(key)
            return _preview_cache[key]
    try:
        data = compose(d, lines, maxw)
    except Err:
        raise
    except Exception as exc:
        raise Err(500, 'Could not composite: %s' % exc)
    with _preview_lock:
        _preview_cache[key] = data
        while len(_preview_cache) > 32:
            _preview_cache.popitem(last=False)
    return data


def thumb_bytes(d, stem):
    path = safe_file(d, stem)
    st = os.stat(path)
    key = hashlib.md5(('%s|%s|%s' % (path, st.st_mtime_ns, st.st_size)).encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, key + '.png')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return f.read()
    im = Image.open(path).convert('RGBA')
    im.thumbnail((160, 160))
    buf = io.BytesIO()
    im.save(buf, 'PNG')
    data = buf.getvalue()
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, 'wb') as f:
            f.write(data)
    except OSError:
        pass
    return data


# ---------------------------------------------------------------------- outbox

def _has_file(line):
    return bool(parse_line(line).file)


def read_blocks(d):
    path = os.path.join(d, INFO_NAME)
    lines = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf8') as f:
            lines = f.read().splitlines()
    blocks, cur = [], []
    for ln in lines:
        if not ln.strip():
            if any(_has_file(l) for l in cur):
                blocks.append(cur)
                cur = []
            continue
        cur.append(ln)
    if cur:
        if any(_has_file(l) for l in cur):
            blocks.append(cur)
        elif blocks:
            blocks[-1].extend(cur)  # keep trailing comments
    return blocks


def write_blocks(d, blocks):
    with open(os.path.join(d, INFO_NAME), 'w', encoding='utf8') as f:
        f.write('\n\n'.join('\n'.join(b) for b in blocks) + ('\n' if blocks else ''))


def outbox_view(blocks):
    return {'stacks': [
        [l.split(COMMENT_PREFIX, 1)[0].strip() for l in b if _has_file(l)]
        for b in blocks
    ]}


def outbox_op(d, body):
    blocks = read_blocks(d)
    op = body.get('op')
    if op == 'add':
        lines = [str(l) for l in body.get('lines', []) if _has_file(str(l))]
        for l in lines:
            safe_file(d, parse_line(l).file)
        if lines:
            blocks.append(lines)
    elif op == 'del':
        i = int(body['i'])
        if 0 <= i < len(blocks):
            blocks.pop(i)
    elif op == 'move':
        i, j = int(body['i']), int(body['i']) + int(body['d'])
        if 0 <= i < len(blocks) and 0 <= j < len(blocks):
            blocks[i], blocks[j] = blocks[j], blocks[i]
    else:
        raise Err(400, 'Unknown op')
    write_blocks(d, blocks)
    return outbox_view(blocks)


# ------------------------------------------------------------------------ http

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, cache=None):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        if cache:
            self.send_header('Cache-Control', cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode(), 'application/json')

    def _handle(self, post):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query, keep_blank_values=True).items()}
        try:
            if u.path == '/' and not post:
                return self._send(200, PAGE.encode('utf8'), 'text/html; charset=utf-8')
            d = resolve_dir(q.get('dir', ''))
            if u.path == '/api/list':
                return self._json({'dir': d, 'layers': list_layers(d), 'dirs': list_dirs()})
            if u.path == '/api/outbox':
                if post:
                    n = int(self.headers.get('Content-Length') or 0)
                    return self._json(outbox_op(d, json.loads(self.rfile.read(n) or b'{}')))
                return self._json(outbox_view(read_blocks(d)))
            if u.path == '/thumb':
                return self._send(200, thumb_bytes(d, q.get('f', '')), 'image/png', 'max-age=60')
            if u.path == '/preview':
                lines = [l for l in q.get('lines', '').split('\n') if l.strip()]
                maxw = int(q['w']) if q.get('w', '').isdigit() else None
                return self._send(200, preview_bytes(d, lines, maxw), 'image/png', 'no-cache')
            raise Err(404, 'Not found')
        except Err as e:
            self._send(e.code, str(e).encode('utf8'), 'text/plain; charset=utf-8')
        except Exception as e:
            self._send(500, ('Server error: %s' % e).encode('utf8'), 'text/plain; charset=utf-8')

    def do_GET(self):
        self._handle(False)

    def do_POST(self):
        self._handle(True)


# ------------------------------------------------------------------------ page

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>merge</title>
<style>
:root{--bg:#e8ebee;--panel:#f6f7f8;--line:#cdd2d7;--fg:#1d2328;--mut:#68727c;--acc:#0e6a68;--accbg:#d3ebe9;--warn:#a2331f}
@media (prefers-color-scheme:dark){:root{--bg:#1a1e22;--panel:#23282d;--line:#394046;--fg:#e4e8eb;--mut:#8e98a1;--acc:#52c3bd;--accbg:#1d3b3a;--warn:#ff8a73}}
*{box-sizing:border-box}[hidden]{display:none!important}
html,body{height:100%;margin:0}
body{display:flex;flex-direction:column;background:var(--bg);color:var(--fg);font:13px/1.35 ui-sans-serif,system-ui,"Segoe UI",sans-serif}
header{display:flex;gap:8px;align-items:center;padding:8px 12px;border-bottom:1px solid var(--line);background:var(--panel)}
header b{font-size:14px;margin-right:4px}
#dir{flex:1;min-width:0;font:12px ui-monospace,Menlo,Consolas,monospace;padding:5px 8px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--fg)}
#toast{color:var(--warn);max-width:40%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
button{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:4px 8px;cursor:pointer}
button:hover{border-color:var(--acc)}
button:focus-visible,input:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
button.pri{background:var(--acc);border-color:var(--acc);color:var(--panel);font-weight:600}
main{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 430px 270px}
section{min-height:0;display:flex;flex-direction:column;border-left:1px solid var(--line);background:var(--panel)}
#pv{border-left:0;background:var(--bg)}
.bar{display:flex;gap:8px;align-items:center;justify-content:space-between;padding:8px;border-bottom:1px solid var(--line)}
#stage{flex:1;min-height:0;position:relative;margin:12px;border:1px solid var(--line);
 background:repeating-conic-gradient(#0000000f 0 25%,#0000 0 50%) 0 0/16px 16px}
#img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
#msg{position:absolute;inset:0;display:grid;place-items:center;color:var(--mut);padding:20px;text-align:center}
#info{padding:0 14px 10px;color:var(--mut)}
#layers,#obs{flex:1;overflow-y:auto;min-height:0}
#picker.locked #layers,#picker.locked #send{opacity:.4;pointer-events:none}
.row{display:flex;gap:8px;padding:6px 8px;border-bottom:1px solid var(--line)}
.row.off .body{opacity:.4}
.ctl{display:grid;grid-template-columns:auto auto;gap:3px;align-content:start}
.ctl input{grid-column:1/3;margin:2px 0 2px 2px}
.ctl button{padding:1px 0;width:24px;font-size:11px;line-height:18px}
.ctl .mu.on{background:var(--acc);color:var(--panel);border-color:var(--acc)}
.body{flex:1;min-width:0}
.name{font:11px ui-monospace,Menlo,Consolas,monospace;color:var(--mut);margin-bottom:3px}
.vars{display:flex;flex-wrap:wrap;gap:3px;max-height:158px;overflow-y:auto}
.th{padding:2px;width:64px;display:flex;flex-direction:column;align-items:center;gap:1px;font-size:10px}
.th img{width:58px;height:58px;object-fit:contain;background:repeating-conic-gradient(#00000014 0 25%,#0000 0 50%) 0 0/10px 10px}
.th span{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.th.sel{border-color:var(--acc);background:var(--accbg);box-shadow:inset 0 0 0 1px var(--acc)}
.ob{display:grid;grid-template-columns:72px 1fr auto;gap:8px;padding:8px;border-bottom:1px solid var(--line);cursor:pointer;align-items:start}
.ob.act{background:var(--accbg);box-shadow:inset 3px 0 0 var(--acc)}
.ob img{width:72px;height:72px;object-fit:contain;background:repeating-conic-gradient(#00000014 0 25%,#0000 0 50%) 0 0/10px 10px}
.olines{font:10.5px/1.4 ui-monospace,Menlo,Consolas,monospace;white-space:pre;overflow:hidden;max-height:72px;color:var(--mut)}
.octl{display:flex;flex-direction:column;gap:3px}
.octl button{padding:0 6px;font-size:11px}
.empty{color:var(--mut);padding:14px}
@media (max-width:900px){main{grid-template-columns:1fr;grid-auto-rows:60vh;overflow-y:auto}}
</style></head><body>
<header>
 <b>merge</b>
 <input id="dir" list="dirs" spellcheck="false" aria-label="Working directory">
 <datalist id="dirs"></datalist>
 <button id="load">Load folder</button>
 <span id="toast" role="status"></span>
</header>
<main>
 <section id="pv">
  <div id="stage"><img id="img" alt="Composite preview" hidden><div id="msg"></div></div>
  <div id="info"></div>
 </section>
 <section id="picker">
  <div class="bar"><button id="send" class="pri">Send to outbox</button><span id="count"></span></div>
  <div id="layers"></div>
 </section>
 <section id="outbox">
  <div class="bar"><b>Outbox</b><button id="back" hidden>Back to workspace</button></div>
  <div id="obs"></div>
 </section>
</main>
<script>
const $=s=>document.querySelector(s), q=o=>new URLSearchParams(o).toString();
const S={dir:'',layers:[],stacks:[],active:-1};

function h(tag,props={},...kids){
  const e=document.createElement(tag);
  for(const [k,v] of Object.entries(props)){
    if(k==='on') for(const [ev,f] of Object.entries(v)) e.addEventListener(ev,f);
    else if(k==='class') e.className=v; else e[k]=v;
  }
  e.append(...kids); return e;
}
async function api(url,body){
  const r=await fetch(url,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:undefined);
  const t=await r.text(); if(!r.ok) throw new Error(t); return JSON.parse(t);
}
function toast(m){const t=$('#toast');t.textContent=m;t.title=m;clearTimeout(toast.h);toast.h=setTimeout(()=>t.textContent='',5000)}

const lineOf=l=>l.variants[l.sel][0]+(l.mult?'*':'');
const wsLines=()=>S.layers.filter(l=>l.vis).map(lineOf);

async function load(dir){
  try{
    const d=await api('/api/list?'+q({dir}));
    S.dir=d.dir; $('#dir').value=d.dir;
    S.layers=d.layers.map(l=>({...l,sel:0,vis:true,mult:false}));
    $('#dirs').replaceChildren(...d.dirs.map(x=>h('option',{value:x})));
    S.stacks=(await api('/api/outbox?'+q({dir:S.dir}))).stacks;
    S.active=-1; renderPicker(); sync();
    if(!d.layers.length) toast('No .png files in this folder');
  }catch(e){toast(e.message);$('#dir').value=S.dir}
}

function count(){$('#count').textContent=S.layers.filter(l=>l.vis).length+' of '+S.layers.length+' layers shown'}

function renderPicker(){
  const box=$('#layers'), top=box.scrollTop; box.replaceChildren();
  S.layers.forEach((l,i)=>{
    const row=h('div',{class:'row'+(l.vis?'':' off')});
    const cb=h('input',{type:'checkbox',checked:l.vis,title:'Show or hide layer',on:{change:()=>{l.vis=cb.checked;row.classList.toggle('off',!l.vis);count();schedule()}}});
    const up=h('button',{textContent:'▲',title:'Move layer up',on:{click:()=>move(i,-1)}});
    const dn=h('button',{textContent:'▼',title:'Move layer down',on:{click:()=>move(i,1)}});
    const mu=h('button',{class:'mu'+(l.mult?' on':''),textContent:'×',title:'Multiply blend (writes * suffix)',on:{click:()=>{l.mult=!l.mult;mu.classList.toggle('on',l.mult);schedule()}}});
    const vs=h('div',{class:'vars'});
    l.variants.forEach(([stem,label],j)=>{
      vs.append(h('button',{class:'th'+(j===l.sel?' sel':''),title:stem,on:{click:()=>{
        l.sel=j; vs.querySelectorAll('.th').forEach((x,k)=>x.classList.toggle('sel',k===j)); schedule();
      }}},h('img',{src:'/thumb?'+q({dir:S.dir,f:stem}),loading:'lazy',alt:''}),h('span',{textContent:label})));
    });
    row.append(h('div',{class:'ctl'},cb,up,dn,mu),h('div',{class:'body'},h('div',{class:'name',textContent:l.name}),vs));
    box.append(row);
  });
  box.scrollTop=top; count();
}
function move(i,d){
  const j=i+d; if(j<0||j>=S.layers.length) return;
  [S.layers[i],S.layers[j]]=[S.layers[j],S.layers[i]]; renderPicker(); schedule();
}

async function ob(body){S.stacks=(await api('/api/outbox?'+q({dir:S.dir}),body)).stacks}
function cbtn(label,title,fn){return h('button',{textContent:label,title,on:{click:e=>{e.stopPropagation();fn().catch(x=>toast(x.message))}}})}
function renderOutbox(){
  const box=$('#obs'); box.replaceChildren();
  if(!S.stacks.length) box.append(h('p',{class:'empty',textContent:'Nothing here yet. Choose variants, then send the composition to the outbox.'}));
  S.stacks.forEach((lines,i)=>{
    box.append(h('div',{class:'ob'+(i===S.active?' act':''),on:{click:()=>select(i===S.active?-1:i)}},
      h('img',{src:'/preview?'+q({dir:S.dir,w:200,lines:lines.join('\n')}),loading:'lazy',alt:''}),
      h('div',{class:'olines',textContent:lines.join('\n')}),
      h('div',{class:'octl'},
        cbtn('▲','Move up',()=>shift(i,-1)),cbtn('▼','Move down',()=>shift(i,1)),cbtn('✕','Delete',()=>del(i)))));
  });
}
async function shift(i,d){
  const j=i+d; if(j<0||j>=S.stacks.length) return;
  await ob({op:'move',i,d});
  if(S.active===i) S.active=j; else if(S.active===j) S.active=i;
  sync();
}
async function del(i){
  await ob({op:'del',i});
  if(S.active===i) S.active=-1; else if(S.active>i) S.active--;
  sync();
}
function select(i){S.active=i; sync()}
function sync(){
  renderOutbox();
  $('#picker').classList.toggle('locked',S.active>=0);
  $('#back').hidden=S.active<0;
  schedule();
}

let tok=0,tm,cur;
function schedule(){clearTimeout(tm);tm=setTimeout(preview,40)}
async function preview(){
  const t=++tok, lines=S.active>=0?S.stacks[S.active]:wsLines(), img=$('#img'), msg=$('#msg');
  $('#info').textContent=(S.active>=0?'Outbox item '+(S.active+1):'Workspace')+' · '+lines.length+' layer'+(lines.length===1?'':'s');
  if(!lines.length){img.hidden=true;msg.textContent='No layers shown. Tick a layer to see it here.';return}
  try{
    const r=await fetch('/preview?'+q({dir:S.dir,w:1600,lines:lines.join('\n')}));
    if(!r.ok) throw new Error(await r.text());
    const b=await r.blob(); if(t!==tok) return;
    const u=URL.createObjectURL(b); if(cur) URL.revokeObjectURL(cur); cur=u;
    img.src=u; img.hidden=false; msg.textContent='';
  }catch(e){if(t===tok){img.hidden=true;msg.textContent=e.message}}
}

$('#send').onclick=async()=>{
  const lines=wsLines(); if(!lines.length) return toast('Show at least one layer first');
  try{await ob({op:'add',lines}); sync(); toast('Sent to outbox')}catch(e){toast(e.message)}
};
$('#back').onclick=()=>select(-1);
$('#load').onclick=()=>load($('#dir').value.trim());
$('#dir').addEventListener('keydown',e=>{if(e.key==='Enter')load($('#dir').value.trim())});
load('');
</script></body></html>
'''


def main():
    ap = argparse.ArgumentParser(description='Web UI for compositing layered PNGs')
    ap.add_argument('--host', default='127.0.0.1', help='default=127.0.0.1')
    ap.add_argument('--port', type=int, default=0, help='default=0 (pick a free port)')
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    host = 'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host
    print('Serving %s' % ROOT)
    if not HAVE_BLEND_MODES:
        print('note: numpy/blend_modes not found, using approximate multiply')
    print('Open http://%s:%d/' % (host, server.server_address[1]))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
