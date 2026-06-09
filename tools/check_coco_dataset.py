#!/usr/bin/env python3
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

def infer_root(ann):
    ann=Path(ann); root=ann.parent.parent if ann.parent.name=='annotations' else ann.parent
    return root,[root,root/'train2017',root/'test2017',root/'images',root/'images'/'train2017',root/'images'/'test2017']

def fkey(fn):
    p=Path(fn); nums=re.findall(r'\d+',p.stem); return str(p.parent), int(nums[-1]) if nums else None

def check(name,ann,t):
    ann=Path(ann); lines=['# '+name+' '+ann.name,'- Annotation: '+str(ann)+'','- Exists: '+str(ann.exists())]
    if not ann.exists(): return '\n'.join(lines), False
    data=json.loads(ann.read_text()); imgs=data.get('images',[]); anns=data.get('annotations',[]); cats=data.get('categories',[])
    root,cands=infer_root(ann); sample=[im.get('file_name','') for im in imgs[:100]]
    scored=sorted([(sum((c/f).exists() for f in sample),c) for c in cands], reverse=True, key=lambda x:x[0]); imroot=scored[0][1]
    by=defaultdict(list)
    for a in anns: by[a.get('image_id')].append(a)
    miss=[]; bad=[]; modes=Counter(); empty=0; illegal=0; oob=0; seq=defaultdict(list)
    for im in imgs:
        path=imroot/im.get('file_name','')
        if not path.exists():
            if len(miss)<20: miss.append(str(path))
            w,h=im.get('width',0),im.get('height',0)
        else:
            try:
                with Image.open(path) as img: modes[img.mode]+=1; w,h=img.size
            except Exception as e:
                if len(bad)<20: bad.append(str(path)+': '+str(e))
                w,h=im.get('width',0),im.get('height',0)
        if not by.get(im['id']): empty+=1
        for a in by.get(im['id'],[]):
            x,y,bw,bh=a.get('bbox',[0,0,0,0])
            if bw<=0 or bh<=0: illegal+=1
            if x<0 or y<0 or x+bw>w+1e-3 or y+bh>h+1e-3: oob+=1
        s,i=fkey(im.get('file_name',''))
        if i is not None: seq[s].append(i)
    can=False; ex=[]
    for s,ids in seq.items():
        ids=sorted(set(ids)); run=best=1
        for a,b in zip(ids,ids[1:]):
            run=run+1 if b==a+1 else 1; best=max(best,run)
        can = can or best>=t
        if len(ex)<5: ex.append(s+': frames='+str(len(ids))+', longest_run>='+str(best))
    cat_ids=sorted(c['id'] for c in cats); single=any(m in ('L','I;16','I') for m in modes)
    lines += ['- Inferred image root: '+str(imroot)+'','- Candidate hits: '+str([(str(c),n) for n,c in scored]),'- Images: '+str(len(imgs)),'- Annotations: '+str(len(anns)),'- Categories: '+str(len(cats))+' ids='+str(cat_ids),'- Category ids continuous from 1: '+str(cat_ids==list(range(1,len(cat_ids)+1))),'- Empty-label images: '+str(empty),'- Missing image samples: '+str(len(miss)),'- Bad image samples: '+str(len(bad)),'- Illegal boxes: '+str(illegal),'- Out-of-bound boxes: '+str(oob),'- Image modes: '+str(dict(modes)),'- Single-channel present: '+str(single),'- T='+str(t)+' clip constructible: '+str(can),'- Boundary policy: repeat boundary frames / identity fallback in SMAHE when fewer than T valid predecessors exist.','','## Sequence examples']
    lines += ['- '+x for x in ex] + ['','## Missing samples'] + ['- '+x+'' for x in miss] + ['','## Bad samples'] + ['- '+x+'' for x in bad]
    return '\n'.join(lines), (not miss and not bad and illegal==0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--name',required=True); ap.add_argument('--train-ann',required=True); ap.add_argument('--test-ann',required=True); ap.add_argument('--out',required=True); ap.add_argument('--t',type=int,default=5); a=ap.parse_args()
    parts=[]; ok=True
    for ann in [a.train_ann,a.test_ann]:
        txt,o=check(a.name,ann,a.t); parts.append(txt); ok=ok and o
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text('\n\n'.join(parts)); print('wrote '+a.out+'; ok='+str(ok))
if __name__=='__main__': main()
