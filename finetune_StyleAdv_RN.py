import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.distributed as dist
import random
import time
from options import parse_args
from utils.PSG import PseudoSampleGenerator
from methods.backbone_multiblock import model_dict
from methods.StyleAdv_RN_GNN import StyleAdvGNN
from data.datamgr import get_few_shot_datamgr
from utils.distributed_utils import cleanup_distributed, is_dist_avail_and_initialized, is_main_process, setup_distributed
from utils.path_utils import validate_dataset_root

#Finetune_LR = 0.001
Finetune_LR = 0.001
#the finetuning is very sensitive to lr

def _load_state_flexible(model, state_dict):
    """Load state dict while dropping incompatible or unused keys."""
    current = model.state_dict()
    filtered = {}
    dropped = []
    for k, v in state_dict.items():
        if k not in current or current[k].shape != v.shape:
            dropped.append(k)
            continue
        filtered[k] = v
    if dropped:
        print(f"[finetune load_state_flexible] dropped {len(dropped)} keys, e.g., {dropped[:3]}")
    model.load_state_dict(filtered, strict=False)

def finetune(novel_loader, n_pseudo=75, n_way=5, n_support=5):
    iter_num = len(novel_loader)
    acc_all = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_dir = '%s/checkpoints/%s/best_model.tar' % (params.save_dir, params.resume_dir)
    state = torch.load(checkpoint_dir)['state']
    for ti, (x, global_y) in enumerate(novel_loader):  # x:(5, 20, 3, 224, 224)
        model = StyleAdvGNN(
            model_dict[params.model],
            n_way=n_way,
            n_support=n_support,
            device=device,
            dataset_name=params.testset,
            data_root=params.data_dir,
            use_style_prompt=getattr(params, "use_style_prompt", 0),
            clip_align_weight=getattr(params, "clip_align_weight", 0.0),
            style_prompt_dim=getattr(params, "style_prompt_dim", None),
            clip_model_name=getattr(params, "clip_model_name", "ViT-B/32"),
        ).to(device)
        _load_state_flexible(model, state)
        x = x.to(device)
        global_y = global_y.to(device)

        # Finetune components initialization
        xs = x[:, :n_support].reshape(-1, *x.size()[2:])  # (25, 3, 224, 224)
        pseudo_q_genrator = PseudoSampleGenerator(n_way, n_support, n_pseudo)
        loss_fun = nn.CrossEntropyLoss().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=Finetune_LR)

        # Finetune process
        n_query = n_pseudo // n_way
        pseudo_set_y = torch.from_numpy(np.repeat(range(n_way), n_query)).to(device)
        model.n_query = n_query
        model.train()
        for epoch in range(params.finetune_epoch):
            opt.zero_grad()
            pseudo_set = pseudo_q_genrator.generate(xs)
            pseudo_set = pseudo_set.view(n_way, n_support + n_query, *x.size()[2:])  # reshape back to episode shape
            scores = model.set_forward(pseudo_set, global_y=global_y)  # (5*n_query, 5)
            loss = loss_fun(scores, pseudo_set_y)
            loss.backward()
            opt.step()
            del pseudo_set, scores, loss

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Inference process
        n_query = x.size(1) - n_support
        model.n_query = n_query
        yq = np.repeat(range(n_way), n_query)
        with torch.no_grad():
            scores = model.set_forward(x, global_y=global_y)  # (80, 5)
            _, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()  # (80, 1)
            top1_correct = np.sum(topk_ind[:, 0] == yq)
            acc = top1_correct * 100.0 / (n_way * n_query)
            acc_all.append(acc)
        del scores, topk_labels

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        if is_main_process():
            print('Task %d : %4.2f%%, mean Acc: %4.2f' % (ti, acc, np.mean(np.array(acc_all))))

    if is_dist_avail_and_initialized():
        gathered = [None for _ in range(params.world_size)]
        dist.all_gather_object(gathered, acc_all)
        merged_acc = []
        for shard in gathered:
            merged_acc.extend(shard)
        acc_all = merged_acc

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    if is_main_process():
        print('Test Acc = %4.2f +- %4.2f%%' % (acc_mean, 1.96 * acc_std / np.sqrt(len(acc_all))))

if __name__=='__main__':
    seed = 0
    params = parse_args('train')
    device = setup_distributed(params)
    rank_seed = seed + getattr(params, "rank", 0)
    if is_main_process():
        print("set seed = %d" % seed)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    params.dataset = params.source_dataset
    if not getattr(params, "testset", None):
        params.testset = params.source_dataset

    missing_paths = validate_dataset_root(
        params.data_dir,
        [params.source_dataset, params.testset],
    )
    if missing_paths:
        raise FileNotFoundError(
            "Dataset folders were not found under --data_dir:\n" + missing_paths
        )

    image_size = 224
    iter_num = getattr(params, "n_episodes_test", 1000)
    n_query = 16
    n_pseudo = 75  

    if is_main_process():
        print('Loading target dataset!:', params.testset)
    datamgr = get_few_shot_datamgr(
        params.testset,
        episodic=True,
        image_size=image_size,
        n_way=params.test_n_way,
        n_support=params.n_shot,
        n_query=n_query,
        n_eposide=iter_num,
        data_root=params.data_dir,
        num_workers=params.eval_num_workers,
        pin_memory=bool(getattr(params, "pin_memory", 1)),
        persistent_workers=bool(getattr(params, "persistent_workers", 1)),
        prefetch_factor=params.prefetch_factor,
        world_size=getattr(params, "world_size", 1),
        rank=getattr(params, "rank", 0),
    )
    novel_loader = datamgr.get_data_loader(aug=False)

    start = time.perf_counter()
    finetune(novel_loader, n_pseudo=n_pseudo, n_way=params.test_n_way, n_support=params.n_shot)
    end = time.perf_counter()
    if is_main_process():
        print('Running time: %s Seconds: %s Min: %s Min per epoch'%(end-start, (end-start)/60, (end-start)/60/iter_num))
    cleanup_distributed()
    
