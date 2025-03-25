from yacs.config import CfgNode as CN


def reset_cfg(cfg, args):
    cfg.config_path = args.config_path
    cfg.gpu_id = args.gpu_id
    cfg.boundaries = args.boundaries
    cfg.covars_scale = args.covars_scale
    cfg.k = args.k


def extend_cfg(cfg):
    """
    Add config variables.
    """
    cfg.dataset_root = ""
    cfg.model_backbone_name = ""
    cfg.input_size = (-1, -1)
    cfg.prompt_template = ""
    cfg.dataset = ""
    cfg.seed = -1
    cfg.use_validation = False

    cfg.train_one_dataset = -1
    cfg.zero_shot = False
    cfg.MTIL_order_2 = False
    
    cfg.IAP = CN()
    cfg.IAP.prompt_depth_vision = 1
    cfg.IAP.prompt_depth_text = 1
    cfg.IAP.n_ctx_vision = 12
    cfg.IAP.n_ctx_text = 12
    cfg.IAP.optim = CN()
    cfg.IAP.optim.batch_size = 64
    cfg.IAP.optim.name = "SGD"
    cfg.IAP.optim.lr = 0.05
    cfg.IAP.optim.max_epoch = 10
    cfg.IAP.optim.weight_decay = 0
    cfg.IAP.optim.lr_scheduler = "cosine"
    cfg.IAP.optim.warmup_epoch = 0
    cfg.IAP.batchwise_prompt = False


def setup_cfg(args):
    cfg = CN()
    extend_cfg(cfg)
    cfg.merge_from_file(args.config_path)
    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)
    return cfg


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)