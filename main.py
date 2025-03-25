import argparse
import os
import sys
from IAP.datasets import get_dataset
from IAP.utils import get_transform, set_random_seed
from IAP.setup_cfg import setup_cfg, print_args
from IAP.IAP import IAP



def run_exp(cfg):
    device = [int(s) for s in cfg.gpu_id.split(',')]
    train_dataset, classes_names, templates = get_dataset(cfg, split='train', transforms=get_transform(cfg))
    val_dataset, _, _ = get_dataset(cfg, split='val', transforms=get_transform(cfg))
    eval_dataset, _, _ = get_dataset(cfg, split='test', transforms=get_transform(cfg))
    cfg.nb_task = len(eval_dataset)

    trainer = IAP(cfg, device, classes_names, templates)
    datasets = {'train': train_dataset, 'val': val_dataset, 'test': eval_dataset}
    trainer.train_and_eval(cfg, datasets)

def main(args):
    cfg = setup_cfg(args)
    cfg.command = ' '.join(sys.argv)
    cfg.boundaries = [float(boundary) for boundary in cfg.boundaries.split(",")]
    cfg.log_path = os.path.join('experiments', f'{cfg.dataset}')
    if not os.path.exists(cfg.log_path):
        os.makedirs(cfg.log_path)
    with open(os.path.join(cfg.log_path, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    with open(os.path.join(cfg.log_path, 'output.txt'), 'w') as f:
        pass
    print_args(args, cfg)
    set_random_seed(cfg.seed)
    run_exp(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="configs/MTIL.yaml", help="path to config")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu id")
    parser.add_argument("--boundaries", type=str, default='0.8,0.2', help="boundaries")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
