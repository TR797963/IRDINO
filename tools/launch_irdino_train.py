#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path
import yaml

def updates(bs):
    return [f'train_dataloader.total_batch_size={bs}', f'val_dataloader.total_batch_size={bs}', f'batch_size={bs}', f'per_gpu_batch_size={bs}']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('-c','--config',required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--extra',nargs='*',default=[]); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    batches=[int(cfg.get('per_gpu_batch_size', cfg.get('batch_size', 48)))] + [int(x) for x in cfg.get('oom_fallback_batch_size',[32,24,16,8])]
    seen=[]
    for b in batches:
        if b not in seen: seen.append(b)
    log=Path('logs')/(Path(args.config).stem.replace('irdino_smahe_n_','')+'_oom_fallback.log'); log.parent.mkdir(exist_ok=True)
    for b in seen:
        cmd=[sys.executable,'train.py','-c',args.config,'-d',args.device,'-u',*updates(b),*args.extra]
        print(f'[IRDINO] trying batch_size={b}')
        with log.open('a') as f: f.write('\n[launch] batch_size=%s cmd=%s\n'%(b,' '.join(cmd)))
        p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        with log.open('a') as f: f.write(p.stdout)
        if p.returncode==0:
            print(f'[IRDINO] finished with batch_size={b}'); return 0
        oom=('out of memory' in p.stdout.lower()) or ('cuda oom' in p.stdout.lower())
        if not oom:
            print(p.stdout[-4000:]); return p.returncode
        print(f'[IRDINO] CUDA OOM at batch_size={b}; fallback')
    return 1
if __name__=='__main__': raise SystemExit(main())
