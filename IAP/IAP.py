import os.path as osp
import os
import json
import statistics
from tqdm import tqdm
import sys
from collections import defaultdict

import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .utils import cosine_loss_3d, cal_MTIL_metrics

from continuum.metrics import Logger
from IAP.utils import build_cosine_scheduler
from IAP.datasets import parse_sample

from torch.distributions.multivariate_normal import MultivariateNormal

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, with_ori=False):

    backbone_name = cfg.model_backbone_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")


    design_details = {"vision_depth": cfg.IAP.prompt_depth_vision,
                      "language_depth": cfg.IAP.prompt_depth_text,
                      "vision_ctx": cfg.IAP.n_ctx_vision,
                      "language_ctx": cfg.IAP.n_ctx_text,
                      "pool_size": cfg.nb_task}
    train_model = clip.build_model(state_dict or model.state_dict(), design_details)

    if with_ori:
        design_details = {"vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        ori_model = clip.build_model(state_dict or model.state_dict(), design_details)
        return train_model, ori_model
    return train_model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype


    def forward(self, prompts, tokenized_prompts, indices, batch_weight=None, raw_prompt=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, indices, batch_weight, raw_prompt)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptProcessor(nn.Module):
    def __init__(self, cfg, classnames, templates, clip_model):
        super().__init__()
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.input_size[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if isinstance(classnames[0], list):
            self.n_cls = 0
            self.class_ids_per_task = []
            self.classnames = []
            for idx, cls_name in enumerate(classnames):
                cur_n = len(cls_name)
                self.class_ids_per_task.append([i for i in range(self.n_cls, self.n_cls+cur_n)])
                cls_name = [templates[idx](name) for name in cls_name]
                self.classnames += cls_name
                self.n_cls += cur_n
        else:
            raise NotImplementedError
        self.cur_n_cls = 0


        self.classnames = [name.replace("_", " ") for name in self.classnames]
        self.all_name_lens = [len(_tokenizer.encode(name)) for name in self.classnames]
        all_prompts = [name for name in self.classnames]
        self.register_buffer("all_tokenized_prompts", torch.cat([clip.tokenize(p) for p in all_prompts]))
        with torch.no_grad():
            self.register_buffer("all_embedding", clip_model.token_embedding(self.all_tokenized_prompts).type(clip_model.dtype))
        self.register_buffer("token_prefix", self.all_embedding[:, :1, :])
        self.register_buffer("token_suffix", self.all_embedding[:, 1:, :])
        self.register_buffer("tokenized_prompts", self.all_tokenized_prompts.clone())


    def forward(self, indices):
        batch_size = indices.size(0)
        prefix = self.token_prefix.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        suffix = self.token_suffix.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        prompts = torch.cat([prefix, suffix], dim=2)
        prompts = prompts.view(batch_size*self.cur_n_cls, prompts.size(2), prompts.size(3))
        tokenized_prompts = self.tokenized_prompts.unsqueeze(0).repeat(batch_size, 1, 1).view(batch_size*self.cur_n_cls, -1)  
        return prompts, tokenized_prompts
    

    def update_classnames(self, task_id):
        class_idx = self.class_ids_per_task[task_id]
        class_idx_tensor = torch.tensor(class_idx, dtype=torch.int, device=self.all_embedding.device)
        self.token_prefix = self.all_embedding[class_idx_tensor, :1, :]
        self.token_suffix = self.all_embedding[class_idx_tensor, 1:, :]
        self.tokenized_prompts = self.all_tokenized_prompts[class_idx_tensor]
        self.name_lens = [self.all_name_lens[idx] for idx in class_idx]
        self.cur_n_cls = len(class_idx)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, templates, clip_model, clip_model_ori=None):
        super().__init__()
        self.prompt_processor = PromptProcessor(cfg, classnames, templates, clip_model)
        self.image_encoder = clip_model.visual
        self.image_encoder_ori = clip_model_ori.visual
        self.text_encoder = TextEncoder(clip_model) 
        self.logit_scale = clip_model.logit_scale 
        self.dtype = clip_model.dtype 
        self.vis_dim = clip_model.visual.output_dim
        self.pool_size = cfg.nb_task
        self.visual_prompt = cfg.IAP.prompt_depth_vision > 0
        self.batchwise_prompt = cfg.IAP.batchwise_prompt

        self.class_means = {}
        self.class_covars = {}
        
        self.task_means = {}
        self.task_covars = {}     
        self.task_learnt = torch.tensor(0, dtype=torch.int)
        self.boundaries = cfg.boundaries
        

        self.val_num = 0
        self.right = 0
    def forward(self, image, task_ids=None, val=None, global_task=None):
        res = {}
        batch_weight = None
        text_batch_weight = None
        with torch.no_grad():
            image_features = self.image_encoder_ori(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            res["image_features"] = image_features.detach()

        if task_ids is not None:
            task_ids = task_ids.type(torch.int).to(image.device)
            assert (task_ids == task_ids[0]).all()
            indices = task_ids[0:1]
            indices = indices.unsqueeze(1)
        else:
            task_dists = [MultivariateNormal(self.task_means[i], self.task_covars[i] * 0.5) for i in range(self.task_learnt.item())]
            task_log_probs = torch.vstack([task_dist.log_prob(image_features) for task_dist in task_dists]).t()
            task_topk, task_indices = task_log_probs.topk(k=1, dim=1)
            task_exp_part = task_topk.squeeze(1)/512-1.0 
            batch_weight = torch.sigmoid(task_exp_part) 
            batch_weight[batch_weight < self.boundaries[1]] = 0.0
            batch_weight[batch_weight > self.boundaries[0]] = 1.0

            prompt_id, id_counts = torch.unique(task_indices, return_counts=True, sorted=True)
            _, major_idx = torch.topk(id_counts, k=1)
            indices = prompt_id[major_idx]
            indices = indices.unsqueeze(0)
            task_indices = indices.item()
            
            zero_weight_mask = (batch_weight == 0.0) | (batch_weight == 1.0)
            non_zero_weight_mask = ~zero_weight_mask
            
            if non_zero_weight_mask.any():
                non_zero_image_features = image_features[non_zero_weight_mask]
            
                class_num = len(self.class_means[task_indices])
                class_dists = [MultivariateNormal(self.class_means[task_indices][i], self.class_covars[task_indices][i] * 0.1) \
                            for i in range(class_num)]
                class_log_probs = torch.vstack([class_dist.log_prob(non_zero_image_features) for class_dist in class_dists]).t() 
                class_topk, _ = class_log_probs.topk(k=5, dim=1)
                class_topk_mean = class_topk.mean(dim=-1)
                class_exp_part = class_topk_mean/512-1.0
                class_batch_weight = torch.sigmoid(class_exp_part) 
                batch_weight[non_zero_weight_mask] = class_batch_weight
                
            text_batch_weight = batch_weight.mean(dim=0, keepdim=True).repeat(self.prompt_processor.cur_n_cls) 
            res["text_batch_weight"] = text_batch_weight[0].item()
            res["raw_indices"] = indices
        res["indices"] = indices

        prompts, tokenized_prompts = self.prompt_processor(indices)
        text_features = self.text_encoder(prompts, tokenized_prompts, indices, text_batch_weight)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if self.visual_prompt:
            image_features = self.image_encoder(image.type(self.dtype), indices, batch_weight, res["image_features"])
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if indices.size(0) == 1:
            logits = logit_scale * image_features @ text_features.t()
        else:
            text_features_resize = text_features.view(image_features.size(0), -1, text_features.size(1)) 
            image_features_resize = image_features.unsqueeze(1)
            logits = logit_scale * image_features_resize @ text_features_resize.permute(0, 2, 1)
            logits = logits.squeeze(1)
        res["outputs"] = logits
        return res
    
    def update_classnames(self, task_id):
        self.prompt_processor.update_classnames(task_id)

class IAP:
    def __init__(self, cfg, device, classes_names, templates):
        self.build_model(cfg, device, classes_names, templates)


    def build_model(self, cfg, device, classes_names, templates):
        print(f"Loading CLIP (backbone: {cfg.model_backbone_name})")
        clip_model, clip_model_ori = load_clip_to_cpu(cfg, with_ori=True)

        print("Building custom CLIP")
        model = CustomCLIP(cfg, classes_names, templates, clip_model, clip_model_ori)
        names_to_update = ["prompt_key", "prefix_pool", "gumbel"]

        for name, param in model.named_parameters():
            update_flag = False
            for name_to_update in names_to_update:
                if name_to_update in name:
                    update_flag = True
            if not update_flag:
                param.requires_grad_(False)

        enabled = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        para_log = f"Parameters to be updated: {enabled}"
        clip_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of parameters to be trained: {}".format(clip_parameters))
        f = open(osp.join(cfg.log_path, 'output.txt'), 'a')
        f.write(para_log + '\n')
        f.close()

        self.model = model
        self.devices = device
        self.device = device[0]

        self.model.to(device[0])
        if len(device) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device)
        self.model_wo_dp = self.model.module if len(device) > 1 else self.model


    def save_model(self, cfg, task_id):
        save_dict = {}
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                save_dict[name] = para
        for name, para in self.model.named_buffers():
            if "means" in name or "covars" in name or "task_learnt" in name:
                save_dict[name] = para
        save_dir = os.path.join(cfg.log_path, 'ckpt')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(save_dict, os.path.join(save_dir, f'task_{task_id}.pt'))
    

    def train_and_eval(self, cfg, datasets):
        acc_list = []
        metric_logger = Logger(list_subsets=["train", "test"])
        metric_writer = open(os.path.join(cfg.log_path, 'metrics.json'), 'w')
        if cfg.zero_shot:
            with torch.no_grad():
                for cur_task in tqdm(range(cfg.nb_task)):
                    self.update_classnames(cur_task)
                    eval_loader = self.get_dataloader(cfg, datasets['test'], cur_task, is_train=False)
                    for sample in eval_loader:
                        inputs, targets, task_ids = parse_sample(sample, is_train=False, task_id=cur_task, cfg=cfg)
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        res = self.model(inputs, task_ids)
                        outputs = res["outputs"]
                        metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")
                cur_all_task_acc = metric_logger.accuracy_per_task
                acc_list.append(cur_all_task_acc)
                log = {'acc_per_task': [round(100 * acc_t, 2) for acc_t in cur_all_task_acc]}
                metric_writer.write(json.dumps(log) + '\n')
                metric_writer.flush()
                print(log)
                return
        
        for task_id in range(cfg.nb_task):
            print(f"Training for task {task_id} has started.")
            self.train_one_task(cfg, task_id, datasets, metric_logger)
            if datasets['val']:
                log = f"Load best epoch weight (epoch {self.best_epoch})."
            print(f"Evaluation for task {task_id} has started.")
            self.eval_all(cfg, datasets, metric_logger, metric_writer, acc_list, global_task=task_id)
            
        res = cal_MTIL_metrics(acc_list)
        metric_writer.write(json.dumps(res["transfer"]) + '\n')
        metric_writer.write(json.dumps(res["avg"]) + '\n')
        metric_writer.write(json.dumps(res["last"]) + '\n')
        metric_writer.write(json.dumps(res["results_mean"]) + '\n')
        metric_writer.flush()


    def train_one_task(self, cfg, task_id, datasets, metric_logger):

        train_dataset, val_dataset, eval_dataset = datasets['train'], datasets['val'], datasets['test']
        train_loader = self.get_dataloader(cfg, train_dataset, task_id, is_train=True)
        self.update_classnames(task_id)
        self.model.train()

        per_epoch_steps = len(train_loader)
        print("per_epoch_steps: ", per_epoch_steps)
        if cfg.IAP.optim.name == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.IAP.optim.lr, weight_decay=cfg.IAP.optim.weight_decay)
        else:
            raise NotImplementedError
        if cfg.IAP.optim.lr_scheduler == 'cosine':
            scheduler = build_cosine_scheduler(optimizer, lr=cfg.IAP.optim.lr, total_step=cfg.IAP.optim.max_epoch*per_epoch_steps)
        elif cfg.IAP.optim.lr_scheduler == 'no':
            scheduler = None
        else:
            raise NotImplementedError
        self.best_epoch = -1
        self.best_acc = -1
        all_image_features = torch.empty([0, self.model_wo_dp.vis_dim], dtype=self.model_wo_dp.dtype, device=self.device)
        class_features_dict = defaultdict(list)
        with torch.no_grad():
            for sample in train_loader:
                inputs, label, _ = parse_sample(sample, is_train=False, task_id=task_id, cfg=cfg)
                image_features = self.model_wo_dp.image_encoder_ori(inputs.type(self.model_wo_dp.dtype).to(self.device))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_image_features = torch.cat([all_image_features, image_features], dim=0)
                for i in range(len(label)):
                            label_i = label[i].item()
                            class_features_dict[label_i].append(image_features[i])
                            
            all_image_features = all_image_features.type(torch.float)
            task_mean = all_image_features.mean(dim=0)
            task_delta = all_image_features - task_mean.unsqueeze(0)
            task_covar = (task_delta.t() @ task_delta) / (all_image_features.size(0) - 1)
            task_covar += torch.eye(task_covar.size(0), device=task_covar.device, dtype=torch.float) * 1e-7
            self.model_wo_dp.task_means[task_id] = task_mean
            self.model_wo_dp.task_covars[task_id] = task_covar
        
            if task_id not in self.model_wo_dp.class_means:
                self.model_wo_dp.class_means[task_id] = {}
                self.model_wo_dp.class_covars[task_id] = {}
                
            for label_i in class_features_dict:
                class_features = torch.stack(class_features_dict[label_i], dim=0).float()
                class_mean = class_features.mean(dim=0)
                class_delta = class_features - class_mean.unsqueeze(0)
                class_covar = (class_delta.t() @ class_delta) / (class_features.size(0) - 1)
                class_covar += torch.eye(class_covar.size(0), device=class_covar.device, dtype=torch.float) * 1e-3
                self.model_wo_dp.class_means[task_id][label_i] = class_mean
                self.model_wo_dp.class_covars[task_id][label_i] = class_covar
        self.model_wo_dp.task_learnt += 1

        for epoch in tqdm(range(cfg.IAP.optim.max_epoch)):
            main_loss_tot = 0
            loss_num = 0
            for idx, sample in enumerate(train_loader):
                if scheduler:
                    cur_iter_idx = epoch*per_epoch_steps+idx 
                    scheduler.step(cur_iter_idx)

                inputs, targets, task_ids = parse_sample(sample, is_train=True, task_id=task_id, cfg=cfg)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                res = self.model(inputs, task_ids) 
                outputs = res["outputs"]
                loss_main = F.cross_entropy(outputs, targets)
                loss = loss_main
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                main_loss_tot += loss_main.item()
                loss_num += 1
                metric_logger.add([outputs.detach().cpu().argmax(dim=1), targets.cpu(), task_ids], subset="train")
            
            log = f"\ntask{task_id}_epoch{epoch}:\n"
            log += f"train acc: {metric_logger.online_accuracy}"
            metric_logger.end_epoch()
            f = open(osp.join(cfg.log_path, 'output.txt'), 'a')
            f.write(log + '\n')
            f.close()
            log = f"avg main loss {round(main_loss_tot/loss_num, 5)}"
            f = open(osp.join(cfg.log_path, 'output.txt'), 'a')
            f.write(log + '\n')
            f.close()
            if val_dataset:
                self.model.eval()
                self.update_classnames(task_id)
                val_loader = self.get_dataloader(cfg, val_dataset, task_id, is_train=False)
                cur_right = torch.FloatTensor([0]).to(self.device)
                cur_all = torch.FloatTensor([0]).to(self.device)
                with torch.no_grad():
                    for sample in val_loader:
                        inputs, targets, task_ids = parse_sample(sample, is_train=False, task_id=task_id, cfg=cfg)
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        res = self.model(inputs, task_ids)
                        outputs = res["outputs"]
                        cur_right += torch.sum((outputs.argmax(dim=1)==targets))
                        cur_all += targets.size(0)
                cur_acc = cur_right/cur_all
                if cur_acc > self.best_acc:
                    self.best_epoch = epoch
                    self.best_acc = cur_acc
                    self.save_model(cfg, task_id)
                self.update_classnames(task_id)
                self.model.train()


    def eval_all(self, cfg, datasets, metric_logger, metric_writer, acc_list, global_task=None):
        eval_dataset = datasets['test']
        self.model.eval()

        for cur_task in tqdm(range(cfg.nb_task)):
            self.update_classnames(cur_task)
            eval_loader = self.get_dataloader(cfg, eval_dataset, cur_task, is_train=False)
            self.evaluate(cfg, eval_loader, cur_task, metric_logger, global_task=global_task)
        cur_all_task_acc = metric_logger.accuracy_per_task
        acc_list.append(cur_all_task_acc)
        log = {'acc_per_task': [round(100 * acc_t, 2) for acc_t in cur_all_task_acc]}
        metric_writer.write(json.dumps(log) + '\n')
        metric_writer.flush()
        print(log)
        metric_logger.end_task()

    def evaluate(self, cfg, loader, task_id, metric_logger=None, global_task=None):
        right_num = 0
        sample_num = 0
        with torch.no_grad():
            for sample in loader:
                inputs, targets, task_ids = parse_sample(sample, is_train=False, task_id=task_id, cfg=cfg)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                res = self.model(inputs, val=task_id, global_task=global_task)
                outputs = res["outputs"]
                if metric_logger:
                    metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")
                right_num += torch.sum(outputs.argmax(dim=1) == targets).item()
                sample_num += inputs.size(0)
        return right_num / sample_num

    def get_dataloader(self, cfg, dataset, task_id, is_train):
        batch_size = cfg.IAP.optim.batch_size
        if isinstance(dataset, list):
            if cfg.IAP.batchwise_prompt and (not is_train):
                batch_size *= 2
            loader = DataLoader(dataset[task_id], batch_size=int(batch_size), shuffle=is_train, num_workers=8)
        else:
            raise NotImplementedError
        return loader

    def update_classnames(self, task_id):
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.update_classnames(task_id)
        else:
            self.model.update_classnames(task_id)
