"""
_merge_ui.py: single-file, offline web UI for compositing layered PNGs.

Usage:
    python _merge_ui.py [--host 127.0.0.1] [--port 0]

Put this file in the folder that holds your PNG folders (or the PNGs themselves),
run it, and open the printed URL. Python standard library only: compositing
happens in the browser (canvas), the server just lists and serves files and
maintains the outbox.

Files are grouped into layers by filename: the LAST run of digits plus anything
trailing it is the variant, everything before it is the layer name.
    B1_i000_2p -> layer 'B1_i000_', variant '2p'
Layers are stacked in reverse-alphabetical order in the picker (top = first),
which is the order _merge.py expects in its input file.

The outbox is a single '_info.txt' next to this script, in _merge.py's format.
Files in subfolders are written with a forward-slash path relative to the
script, e.g. 'B001/B1_a000', so run _merge.py from this same folder.
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
INFO_PATH = os.path.join(ROOT, '_info.txt')
MAX_TREE_DEPTH = 8


class Err(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


# ---------------------------------------------------------------- filesystem

def resolve_dir(rel):
    """Resolve a ROOT-relative dir; must be ROOT or a subdirectory of it."""
    full = os.path.realpath(os.path.join(ROOT, rel or '.'))
    if full != ROOT and not full.startswith(ROOT + os.sep):
        raise Err(403, 'Directory must be inside ' + ROOT)
    if not os.path.isdir(full):
        raise Err(404, 'Not a directory: ' + full)
    return full


def rel_of(full):
    r = os.path.relpath(full, ROOT)
    return '' if r == '.' else r.replace(os.sep, '/')


def safe_file(stem):
    """stem is a ROOT-relative path without extension, e.g. 'B001/B1_a000'."""
    if not stem or '\\' in stem or '\0' in stem:
        raise Err(400, 'Bad file name: ' + stem)
    path = os.path.realpath(os.path.join(ROOT, stem + EXT))
    if not path.startswith(ROOT + os.sep):
        raise Err(403, 'Outside root: ' + stem)
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


def build_tree():
    """Folder tree under ROOT, pruned to folders that (transitively) hold PNGs."""
    def walk(path, rel, depth):
        try:
            names = sorted(os.listdir(path), key=natkey)
        except OSError:
            return None
        pngs = sum(1 for n in names if n.endswith(EXT) and not n.startswith('.'))
        kids = []
        if depth < MAX_TREE_DEPTH:
            for n in names:
                p = os.path.join(path, n)
                if n.startswith('.') or os.path.islink(p) or not os.path.isdir(p):
                    continue
                k = walk(p, rel + '/' + n if rel else n, depth + 1)
                if k:
                    kids.append(k)
        if rel and not pngs and not kids:
            return None
        return {'name': os.path.basename(path), 'rel': rel, 'n': pngs, 'kids': kids}
    return walk(ROOT, '', 0)


# ---------------------------------------------------------------------- outbox

def _strip(line):
    return line.split('#', 1)[0].strip()


def _file_of(line):
    """File stem of an _info.txt line (same notation as _merge.py), or ''."""
    return _strip(line).replace('*', '').split('@', 1)[0].strip()


def read_blocks():
    lines = []
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH, 'r', encoding='utf8') as f:
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


def write_blocks(blocks):
    with open(INFO_PATH, 'w', encoding='utf8') as f:
        f.write('\n\n'.join('\n'.join(b) for b in blocks) + ('\n' if blocks else ''))


def outbox_view(blocks):
    return {'stacks': [[_strip(l) for l in b if _file_of(l)] for b in blocks]}


def outbox_op(body):
    blocks = read_blocks()
    op = body.get('op')
    if op == 'add':
        lines = [str(l) for l in body.get('lines', []) if _file_of(str(l))]
        for l in lines:
            safe_file(_file_of(l))
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
    write_blocks(blocks)
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
            if u.path == '/api/list':
                d = resolve_dir(q.get('dir', ''))
                return self._json({'rel': rel_of(d), 'abs': d, 'layers': list_layers(d)})
            if u.path == '/api/tree':
                return self._json(build_tree())
            if u.path == '/api/outbox':
                if post:
                    n = int(self.headers.get('Content-Length') or 0)
                    return self._json(outbox_op(json.loads(self.rfile.read(n) or b'{}')))
                return self._json(outbox_view(read_blocks()))
            if u.path == '/file':
                with open(safe_file(q.get('f', '')), 'rb') as f:
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
:root{--bg:#e8ebee;--panel:#f6f7f8;--line:#cdd2d7;--fg:#1d2328;--mut:#68727c;--acc:#0e6a68;--accbg:#d3ebe9;--warn:#a2331f;--ob:#6b3fa0;--obbg:#e7ddf4}
@media (prefers-color-scheme:dark){:root{--bg:#1a1e22;--panel:#23282d;--line:#394046;--fg:#e4e8eb;--mut:#8e98a1;--acc:#52c3bd;--accbg:#1d3b3a;--warn:#ff8a73;--ob:#b794f0;--obbg:#31234b}}
*{box-sizing:border-box}[hidden]{display:none!important}
html,body{height:100%;margin:0}
body{display:flex;flex-direction:column;background:var(--bg);color:var(--fg);font:14px/1.35 ui-sans-serif,system-ui,"Segoe UI",sans-serif}
header{position:relative;display:flex;gap:10px;align-items:center;padding:10px 14px;border-bottom:1px solid var(--line);background:var(--panel)}
header b{font-size:15px;margin-right:4px}
#dir{flex:1;min-width:0;height:36px;cursor:pointer;font:13px ui-monospace,Menlo,Consolas,monospace;padding:0 10px;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--fg)}
#toast{color:var(--warn);max-width:35%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#tree{position:absolute;top:100%;left:14px;z-index:10;min-width:300px;max-width:calc(100vw - 28px);max-height:65vh;overflow:auto;padding:4px;
 background:var(--panel);border:1px solid var(--line);border-radius:6px;box-shadow:0 8px 24px #0004}
.tn{display:flex;width:100%;justify-content:space-between;gap:14px;align-items:center;text-align:left;border:0;background:none;border-radius:4px;min-height:34px;font:13px ui-monospace,Menlo,Consolas,monospace}
.tn:hover{background:var(--accbg)}
.tn.cur{background:var(--accbg);color:var(--acc);font-weight:600}
.tn small{color:var(--mut);font:11px ui-sans-serif,system-ui,sans-serif}
button{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:5px;min-height:36px;min-width:36px;padding:0 12px;cursor:pointer}
button:hover{border-color:var(--acc)}
button:focus-visible,input:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
button.pri{background:var(--acc);border-color:var(--acc);color:var(--panel);font-weight:600;min-height:40px;padding:0 18px}
main{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 440px 290px}
section{min-height:0;display:flex;flex-direction:column;border-left:1px solid var(--line);background:var(--panel)}
#pv{border-left:0;background:var(--bg)}
.bar{display:flex;gap:10px;align-items:center;justify-content:space-between;padding:10px;height:62px;flex:none;border-bottom:1px solid var(--line)}
.grp{display:flex;gap:8px;align-items:center;flex:1;min-width:0}
#all{width:22px;height:22px;margin:0 0 0 2px;accent-color:var(--acc)}
#flt{flex:1;min-width:0;height:36px;font:13px ui-monospace,Menlo,Consolas,monospace;padding:0 10px;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--fg)}
#flt.bad{border-color:var(--warn);outline-color:var(--warn)}
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
.fn{min-height:30px;padding:0 9px;font:12.5px ui-monospace,Menlo,Consolas,monospace}
.fn.sel{border-color:var(--acc);background:var(--accbg);box-shadow:inset 0 0 0 1px var(--acc);font-weight:600}
.ob{display:grid;grid-template-columns:76px 1fr auto;gap:10px;padding:10px;border-bottom:1px solid var(--line);cursor:pointer;align-items:start}
.ob.act{background:var(--obbg);box-shadow:inset 3px 0 0 var(--ob)}
#outbox .bar b{color:var(--ob)}
#outbox button:hover{border-color:var(--ob)}
#outbox button:focus-visible{outline-color:var(--ob)}
button.pri.purple{background:var(--ob);border-color:var(--ob)}
.ob canvas{width:76px;height:76px;object-fit:contain;background:repeating-conic-gradient(#00000014 0 25%,#0000 0 50%) 0 0/10px 10px}
.olines{font:11px/1.4 ui-monospace,Menlo,Consolas,monospace;white-space:pre;overflow:hidden;max-height:76px;color:var(--mut)}
.octl{display:flex;flex-direction:column;gap:4px}
.octl button{min-height:30px;padding:0 10px}
.empty{color:var(--mut);padding:16px}
@media (max-width:900px){main{grid-template-columns:1fr;grid-auto-rows:60vh;overflow-y:auto}}
</style></head><body>
<header>
 <b>merge</b>
 <input id="dir" readonly aria-label="Working folder (click to browse)" title="Click to browse folders">
 <button id="browse" aria-label="Browse folders">Folders ▾</button>
 <span id="toast" role="status"></span>
 <div id="tree" hidden></div>
</header>
<main>
 <section id="pv">
  <div id="stage"><canvas id="cv" hidden></canvas><div id="msg"></div></div>
  <div id="info"></div>
 </section>
 <section id="picker">
  <div class="bar">
   <div class="grp">
    <input type="checkbox" id="all" title="Show or hide all listed layers">
    <input id="flt" placeholder="Filter layers (regex)" spellcheck="false" aria-label="Filter layers by regex">
    <button id="clr" title="Clear filter">✕</button>
   </div>
   <button id="send" class="pri">Send to outbox</button>
  </div>
  <div id="layers"></div>
 </section>
 <section id="outbox">
  <div class="bar"><b>Outbox</b><button id="back" hidden>Back to workspace</button></div>
  <div id="obs"></div>
 </section>
</main>
<script>
const $=s=>document.querySelector(s), q=o=>new URLSearchParams(o).toString();
const S={rel:'',abs:'',layers:[],stacks:[],active:-1,re:null};
const pre=()=>S.rel?S.rel+'/':'';          // folder prefix for _info.txt lines

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
function loadImg(path){
  if(!imgCache.has(path)) imgCache.set(path,new Promise((res,rej)=>{
    const im=new Image();
    im.onload=()=>res(im);
    im.onerror=()=>{imgCache.delete(path);rej(new Error('Could not load '+path+'.png'))};
    im.src='/file?'+q({f:path});
  }));
  return imgCache.get(path);
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

/* ---- folders ---- */
async function load(rel){
  try{
    const d=await api('/api/list?'+q({dir:rel}));
    S.rel=d.rel; S.abs=d.abs; $('#dir').value=d.abs;
    S.layers=d.layers.map(l=>({...l,sel:0,vis:true}));
    S.active=-1; renderPicker(); sync();
    if(!d.layers.length) toast('No .png files in this folder');
  }catch(e){toast(e.message)}
}
function drawTree(n,depth,box){
  const b=h('button',{class:'tn'+(n.rel===S.rel?' cur':''),on:{click:()=>{$('#tree').hidden=true;load(n.rel)}}},
    h('span',{textContent:(depth?'└ ':'')+n.name+(depth?'':'/')}),
    h('small',{textContent:n.n?n.n+' png':''}));
  b.style.paddingLeft=(10+depth*16)+'px';
  box.append(b); n.kids.forEach(k=>drawTree(k,depth+1,box));
}
async function toggleTree(){
  const t=$('#tree');
  if(!t.hidden){t.hidden=true;return}
  try{const root=await api('/api/tree'); t.replaceChildren(); drawTree(root,0,t); t.hidden=false}
  catch(e){toast(e.message)}
}
$('#dir').onclick=$('#browse').onclick=toggleTree;
document.addEventListener('click',e=>{if(!e.target.closest('#tree,#dir,#browse')) $('#tree').hidden=true});
document.addEventListener('keydown',e=>{if(e.key==='Escape') $('#tree').hidden=true});

/* ---- workspace ---- */
const wsLines=()=>S.layers.filter(l=>l.vis).map(l=>pre()+l.variants[l.sel][0]);
const match=l=>!S.re||S.re.test(l.name);
function count(){
  const listed=S.layers.filter(match), n=listed.filter(l=>l.vis).length, all=$('#all');
  all.checked=n>0&&n===listed.length; all.indeterminate=n>0&&n<listed.length;
}
$('#all').onchange=()=>{S.layers.filter(match).forEach(l=>l.vis=$('#all').checked);renderPicker();schedule()};

function renderPicker(){
  const box=$('#layers'), top=box.scrollTop; box.replaceChildren();
  S.layers.forEach((l,i)=>{
    if(!match(l)) return;
    const row=h('div',{class:'row'+(l.vis?'':' off')});
    const cb=h('input',{type:'checkbox',checked:l.vis,title:'Show or hide layer',on:{change:()=>{l.vis=cb.checked;row.classList.toggle('off',!l.vis);count();schedule()}}});
    const vs=h('div',{class:'vars'});
    l.variants.forEach(([stem],j)=>{
      vs.append(h('button',{class:'fn'+(j===l.sel?' sel':''),textContent:stem,title:stem,on:{click:()=>{
        l.sel=j; vs.querySelectorAll('.fn').forEach((x,k)=>x.classList.toggle('sel',k===j)); schedule();
      }}}));
    });
    row.append(
      h('div',{class:'head'},cb,
        h('span',{class:'name',textContent:l.name,title:l.name}),
        h('button',{textContent:'▲',title:'Move layer up',on:{click:()=>move(i,-1)}}),
        h('button',{textContent:'▼',title:'Move layer down',on:{click:()=>move(i,1)}})),
      vs);
    box.append(row);
  });
  box.scrollTop=top; count();
}
function move(i,d){
  let j=i+d; while(j>=0&&j<S.layers.length&&!match(S.layers[j])) j+=d;
  if(j<0||j>=S.layers.length) return;
  [S.layers[i],S.layers[j]]=[S.layers[j],S.layers[i]]; renderPicker(); schedule();
}

/* ---- outbox (single _info.txt, shared across folders) ---- */
async function ob(body){S.stacks=(await api('/api/outbox',body)).stacks}
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
  const send=$('#send');
  send.textContent=S.active>=0?'Load from outbox':'Send to outbox';
  send.classList.toggle('purple',S.active>=0);
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

/* Replace the picker state with the selected outbox entry. */
async function loadFromOutbox(){
  const specs=(S.stacks[S.active]||[]).map(parseLine).filter(s=>s.file).map(s=>{
    const k=s.file.lastIndexOf('/');
    return {...s,dir:k<0?'':s.file.slice(0,k),stem:s.file.slice(k+1)};
  });
  if(!specs.length) return;
  const dir=specs[0].dir;                       // the entry's top layer decides the folder
  await load(dir);                              // fresh state: default order, all shown, first variants
  if(S.rel!==dir) return;
  let skipped=0, dropped=false; const order=[];
  specs.forEach(s=>{
    if(s.mult||s.off) dropped=true;
    const li=S.layers.findIndex(l=>l.variants.some(v=>v[0]===s.stem));
    if(s.dir!==dir||li<0||order.includes(li)){skipped++;return}
    S.layers[li].sel=S.layers[li].variants.findIndex(v=>v[0]===s.stem);
    order.push(li);
  });
  S.layers.forEach(l=>l.vis=false);
  order.forEach(li=>S.layers[li].vis=true);
  const slots=[...order].sort((a,b)=>a-b), picked=order.map(li=>S.layers[li]), out=[...S.layers];
  slots.forEach((slot,k)=>out[slot]=picked[k]);   // entry's layer order, hidden layers keep their slots
  S.layers=out; S.active=-1; renderPicker(); sync();
  const notes=[];
  if(skipped) notes.push(skipped+' line'+(skipped===1?'':'s')+' skipped');
  if(dropped) notes.push('multiply/offset not editable here');
  toast(notes.length?'Loaded; '+notes.join(', '):'Loaded from outbox');
}
$('#send').onclick=async()=>{
  if(S.active>=0) return loadFromOutbox();
  const lines=wsLines(); if(!lines.length) return toast('Show at least one layer first');
  try{await ob({op:'add',lines}); sync(); toast('Sent to outbox')}catch(e){toast(e.message)}
};
$('#flt').oninput=()=>{
  const el=$('#flt');
  try{S.re=el.value?new RegExp(el.value,'i'):null; el.classList.remove('bad')}
  catch{el.classList.add('bad'); return}
  renderPicker();
};
$('#clr').onclick=()=>{$('#flt').value=''; $('#flt').oninput(); $('#flt').focus()};
$('#back').onclick=()=>select(-1);

(async()=>{
  try{S.stacks=(await api('/api/outbox')).stacks}catch(e){toast(e.message)}
  await load('');
})();
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