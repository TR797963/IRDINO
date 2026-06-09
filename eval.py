import argparse
from train import main
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('-c','--config',required=True); p.add_argument('-r','--resume',required=True); p.add_argument('-d','--device',default=None)
    args, rest = p.parse_known_args()
    ns=argparse.Namespace(config=args.config,resume=args.resume,tuning=None,device=args.device,seed=0,use_amp=False,output_dir=None,summary_dir=None,test_only=True,update=rest or None,print_method='builtin',print_rank=0,local_rank=None)
    main(ns)
