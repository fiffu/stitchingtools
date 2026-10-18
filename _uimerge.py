"""
_merge_ui.py: single-file, offline web UI for compositing layered PNGs.

Usage:
    python _merge_ui.py [--host 127.0.0.1] [--port 0]

Put this file next to your PNGs (or in a parent of the folders that hold them),
run it, and open the printed URL. Python standard library only: compositing
happens in the browser (canvas), the server just lists and serves files and
maintains the outbox.

Files are grouped into layers by filename: the LAST run of digits plus anything
trailing it is the variant, everything before it is the layer name.
    B1_i000_2p -> layer 'B1_i000_', variant '2p'
Layers are stacked in reverse-alphabetical order in the picker (top = first),
which is the order _merge.py expects in its input file.

The outbox is '_info.txt' in the loaded directory, in _merge.py's format.
Existing '*' (multiply) and '@x,y' (offset) notation is preserved and rendered.
"""

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.realpath(__file__))
EXT = '.png'
INFO_NAME = '_info.txt'


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


# ---------------------------------------------------------------------- outbox

def _strip(line):
    return line.split('#', 1)[0].strip()


def _file_of(line):
    """File stem of an _info.txt line (same notation as _merge.py), or ''."""
    return _strip(line).replace('*', '').split('@', 1)[0].strip()


def read_blocks(d):
    path = os.path.join(d, INFO_NAME)
    lines = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf8') as f:
            lines = f.read().splitlines()
    blocks, cur = [], []
    for ln in lines:
        if not ln.strip():
            if any(_file_of(l) for l in cur):
                blocks.append(cur)
                cur = []
            continue
        cur.append(ln)
    if cur:
        if any(_file_of(l) for l in cur):
            blocks.append(cur)
        elif blocks:
            blocks[-1].extend(cur)  # keep trailing comments
    return blocks


def write_blocks(d, blocks):
    with open(os.path.join(d, INFO_NAME), 'w', encoding='utf8') as f:
        f.write('\n\n'.join('\n'.join(b) for b in blocks) + ('\n' if blocks else ''))


def outbox_view(blocks):
    return {'stacks': [[_strip(l) for l in b if _file_of(l)] for b in blocks]}


def outbox_op(d, body):
    blocks = read_blocks(d)
    op = body.get('op')
    if op == 'add':
        lines = [str(l) for l in body.get('lines', []) if _file_of(str(l))]
        for l in lines:
            safe_file(d, _file_of(l))
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
            if u.path == '/file':
                with open(safe_file(d, q.get('f', '')), 'rb') as f:
                    return self._send(200, f.read(), 'image/png', 'no-cache')
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
body{display:flex;flex-direction:column;background:var(--bg);color:var(--fg);font:14px/1.35 ui-sans-serif,system-ui,"Segoe UI",sans-serif}
header{display:flex;gap:10px;align-items:center;padding:10px 14px;border-bottom:1px solid var(--line);background:var(--panel)}
header b{font-size:15px;margin-right:4px}
#dir{flex:1;min-width:0;height:36px;font:13px ui-monospace,Menlo,Consolas,monospace;padding:0 10px;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--fg)}
#toast{color:var(--warn);max-width:35%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
button{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:5px;min-height:36px;min-width:36px;padding:0 12px;cursor:pointer}
button:hover{border-color:var(--acc)}
button:focus-visible,input:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
button.pri{background:var(--acc);border-color:var(--acc);color:var(--panel);font-weight:600;min-height:40px;padding:0 18px}
main{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 440px 290px}
section{min-height:0;display:flex;flex-direction:column;border-left:1px solid var(--line);background:var(--panel)}
#pv{border-left:0;background:var(--bg)}
.bar{display:flex;gap:10px;align-items:center;justify-content:space-between;padding:10px;border-bottom:1px solid var(--line)}
#stage{flex:1;min-height:0;position:relative;margin:14px;border:1px solid var(--line);
 background:repeating-conic-gradient(#0000000f 0 25%,#0000 0 50%) 0 0/16px 16px}
#cv{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
#msg{position:absolute;inset:0;display:grid;place-items:center;color:var(--mut);padding:20px;text-align:center}
#info{padding:0 16px 12px;color:var(--mut)}
#layers,#obs{flex:1;overflow-y:auto;min-height:0}
#picker.locked #layers,#picker.locked .grp{opacity:.4;pointer-events:none}
.row{padding:8px 10px;border-bottom:1px solid var(--line)}
.row.off .vars{opacity:.4}
.head{display:flex;gap:6px;align-items:center;margin-bottom:6px}
.head input{width:22px;height:22px;margin:0 6px 0 2px;accent-color:var(--acc)}
.name{flex:1;min-width:0;margin-left:4px;font:600 13px ui-monospace,Menlo,Consolas,monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.vars{display:flex;flex-wrap:wrap;gap:4px}
.grp{display:flex;gap:12px;align-items:center}
#all{width:22px;height:22px;margin:0 0 0 2px;accent-color:var(--acc)}
.fn{min-height:30px;padding:0 9px;font:12.5px ui-monospace,Menlo,Consolas,monospace}
.fn.sel{border-color:var(--acc);background:var(--accbg);box-shadow:inset 0 0 0 1px var(--acc);font-weight:600}
.ob{display:grid;grid-template-columns:76px 1fr auto;gap:10px;padding:10px;border-bottom:1px solid var(--line);cursor:pointer;align-items:start}
.ob.act{background:var(--accbg);box-shadow:inset 3px 0 0 var(--acc)}
.ob canvas{width:76px;height:76px;object-fit:contain;background:repeating-conic-gradient(#00000014 0 25%,#0000 0 50%) 0 0/10px 10px}
.olines{font:11px/1.4 ui-monospace,Menlo,Consolas,monospace;white-space:pre;overflow:hidden;max-height:76px;color:var(--mut)}
.octl{display:flex;flex-direction:column;gap:4px}
.octl button{min-height:30px;padding:0 10px}
.empty{color:var(--mut);padding:16px}
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
  <div id="stage"><canvas id="cv" hidden></canvas><div id="msg"></div></div>
  <div id="info"></div>
 </section>
 <section id="picker">
  <div class="bar"><div class="grp"><input type="checkbox" id="all" title="Show or hide all layers"><button id="send" class="pri">Send to outbox</button></div><span id="count"></span></div>
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

/* ---- compositing (browser canvas; line notation matches _merge.py) ---- */
const imgCache=new Map();
function loadImg(stem){
  const k=S.dir+'\0'+stem;
  if(!imgCache.has(k)) imgCache.set(k,new Promise((res,rej)=>{
    const im=new Image();
    im.onload=()=>res(im);
    im.onerror=()=>{imgCache.delete(k);rej(new Error('Could not load '+stem+'.png'))};
    im.src='/file?'+q({dir:S.dir,f:stem});
  }));
  return imgCache.get(k);
}
function parseLine(line){
  let l=line.split('#',1)[0].trim();
  const mult=l.includes('*'); l=l.replaceAll('*','');
  const at=l.indexOf('@');
  const file=(at<0?l:l.slice(0,at)).trim();
  const off=at<0?null:l.slice(at+1).split(',',2).map(n=>parseInt(n,10));
  return {file,mult,off:off&&off.length===2&&off.every(Number.isFinite)?off:null};
}
/* Draws lines (top-first) into canvas. Resolves true if drawn, false if stale/empty. */
async function render(lines,canvas,maxDim,stale){
  const specs=lines.map(parseLine).filter(s=>s.file).reverse();   // bottom first
  if(!specs.length) return false;
  const imgs=await Promise.all(specs.map(s=>loadImg(s.file)));
  if(stale()) return false;
  const base=imgs[0], W=base.naturalWidth, H=base.naturalHeight;
  const sc=maxDim?Math.min(1,maxDim/Math.max(W,H)):1;
  canvas.width=Math.max(1,Math.round(W*sc)); canvas.height=Math.max(1,Math.round(H*sc));
  const c=canvas.getContext('2d');
  c.imageSmoothingQuality='high'; c.scale(sc,sc);
  specs.forEach((s,i)=>{
    if(i&&s.mult&&s.off) return;                       // unsupported combo, skipped like _merge.py
    c.globalCompositeOperation=(i&&s.mult)?'multiply':'source-over';
    c.drawImage(imgs[i],...(s.off||[0,0]));
  });
  return true;
}

/* ---- workspace ---- */
const wsLines=()=>S.layers.filter(l=>l.vis).map(l=>l.variants[l.sel][0]);

async function load(dir){
  try{
    const d=await api('/api/list?'+q({dir}));
    S.dir=d.dir; $('#dir').value=d.dir;
    S.layers=d.layers.map(l=>({...l,sel:0,vis:true}));
    $('#dirs').replaceChildren(...d.dirs.map(x=>h('option',{value:x})));
    S.stacks=(await api('/api/outbox?'+q({dir:S.dir}))).stacks;
    S.active=-1; renderPicker(); sync();
    if(!d.layers.length) toast('No .png files in this folder');
  }catch(e){toast(e.message);$('#dir').value=S.dir}
}

function count(){
  const n=S.layers.filter(l=>l.vis).length, all=$('#all');
  $('#count').textContent=n+' of '+S.layers.length+' layers shown';
  all.checked=n>0&&n===S.layers.length; all.indeterminate=n>0&&n<S.layers.length;
}
$('#all').onchange=()=>{S.layers.forEach(l=>l.vis=$('#all').checked);renderPicker();schedule()};

function renderPicker(){
  const box=$('#layers'), top=box.scrollTop; box.replaceChildren();
  S.layers.forEach((l,i)=>{
    const row=h('div',{class:'row'+(l.vis?'':' off')});
    const cb=h('input',{type:'checkbox',checked:l.vis,title:'Show or hide layer',on:{change:()=>{l.vis=cb.checked;row.classList.toggle('off',!l.vis);count();schedule()}}});
    const vs=h('div',{class:'vars'});
    l.variants.forEach(([stem,label],j)=>{
      vs.append(h('button',{class:'fn'+(j===l.sel?' sel':''),textContent:stem,title:stem,on:{click:()=>{
        l.sel=j; vs.querySelectorAll('.fn').forEach((x,k)=>x.classList.toggle('sel',k===j)); schedule();
      }}}));
    });
    row.append(
      h('div',{class:'head'},cb,
        h('button',{textContent:'▲',title:'Move layer up',on:{click:()=>move(i,-1)}}),
        h('button',{textContent:'▼',title:'Move layer down',on:{click:()=>move(i,1)}}),
        h('span',{class:'name',textContent:l.name,title:l.name})),
      vs);
    box.append(row);
  });
  box.scrollTop=top; count();
}
function move(i,d){
  const j=i+d; if(j<0||j>=S.layers.length) return;
  [S.layers[i],S.layers[j]]=[S.layers[j],S.layers[i]]; renderPicker(); schedule();
}

/* ---- outbox ---- */
async function ob(body){S.stacks=(await api('/api/outbox?'+q({dir:S.dir}),body)).stacks}
function cbtn(label,title,fn){return h('button',{textContent:label,title,on:{click:e=>{e.stopPropagation();fn().catch(x=>toast(x.message))}}})}
function renderOutbox(){
  const box=$('#obs'); box.replaceChildren();
  if(!S.stacks.length) box.append(h('p',{class:'empty',textContent:'Nothing here yet. Choose variants, then send the composition to the outbox.'}));
  S.stacks.forEach((lines,i)=>{
    const cv=h('canvas');
    box.append(h('div',{class:'ob'+(i===S.active?' act':''),on:{click:()=>select(i===S.active?-1:i)}},
      cv,
      h('div',{class:'olines',textContent:lines.join('\n')}),
      h('div',{class:'octl'},
        cbtn('▲','Move up',()=>shift(i,-1)),cbtn('▼','Move down',()=>shift(i,1)),cbtn('✕','Delete',()=>del(i)))));
    render(lines,cv,160,()=>!cv.isConnected).catch(()=>{});
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

/* ---- preview ---- */
let tok=0,tm;
function schedule(){clearTimeout(tm);tm=setTimeout(preview,20)}
async function preview(){
  const t=++tok, lines=S.active>=0?S.stacks[S.active]:wsLines(), cv=$('#cv'), msg=$('#msg');
  $('#info').textContent=(S.active>=0?'Outbox item '+(S.active+1):'Workspace')+' · '+lines.length+' layer'+(lines.length===1?'':'s');
  if(!lines.length){cv.hidden=true;msg.textContent='No layers shown. Tick a layer to see it here.';return}
  try{
    if(await render(lines,cv,0,()=>t!==tok)){cv.hidden=false;msg.textContent=''}
  }catch(e){if(t===tok){cv.hidden=true;msg.textContent=e.message}}
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
    print('Open http://%s:%d/' % (host, server.server_address[1]))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
