import torch
import torchvision.transforms as transforms
import random

def gamma_correction(x, gamma):
    minv = torch.min(x)
    x = x - minv

    maxv = torch.max(x)
    x = x / maxv

    x = x**gamma
    x = x * maxv + minv
    return x

def random_aug(x):
    #print('x1:', x.size())
    # gamma correction
    if random.random() <= 0.3:
        gamma = random.uniform(1.0, 1.5)
        x = gamma_correction(x, gamma)
    # random erasing with mean value
    mean_v = tuple(x.view(x.size(0), -1).mean(-1))
    re = transforms.RandomErasing(p=0.5, value=mean_v)
    x = re(x)
    # color channel shuffle
    if random.random() <= 0.3:
        l = [0,1,2]
        random.shuffle(l)
        x_c = torch.zeros_like(x)
        x_c[l] = x
        x = x_c
    # horizontal flip or vertical flip
    if random.random() <= 0.5:
        if random.random() <= 0.5:
            x = torch.flip(x, [1])
        else:
            x = torch.flip(x, [2])
    # rotate 90, 180 or 270 degree
    if random.random() <= 0.5:
        degree = [90, 180, 270]
        d = random.choice(degree)
        x = torch.rot90(x, d//90, [1, 2])
    #print('x2:', x.size())
    return x

class PseudoSampleGenerator(object):
    def __init__(self, n_way, n_support, n_pseudo):
        super(PseudoSampleGenerator, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_pseudo = n_pseudo
        self.n_pseudo_per_way = self.n_pseudo//self.n_way

    def generate(self, support_set):  # (5*n_support, 3, 224, 224)
        # 保证“前 n_support = 真实 support”，伪样本全部追加在 support 之后
        n_total, c, h, w = support_set.shape
        assert n_total == self.n_way * self.n_support, "support_set size mismatch"
        support = support_set.view(self.n_way, self.n_support, c, h, w)

        # 生成每类固定数量的 pseudo，保持按类组织
        pseudo_all = []
        if self.n_support <= 5:
            # 重复增强直到每类达到 n_pseudo_per_way
            for w_idx in range(self.n_way):
                per_class = support[w_idx]
                pseudo_list = []
                while len(pseudo_list) < self.n_pseudo_per_way:
                    for s in per_class:
                        pseudo_list.append(random_aug(s))
                        if len(pseudo_list) >= self.n_pseudo_per_way:
                            break
                pseudo_all.append(torch.stack(pseudo_list))
            pseudo = torch.stack(pseudo_all)  # [n_way, n_pseudo_per_way, C, H, W]
        else:
            # 适配 20/50-shot：从每类 support 中采样若干张做增强
            select_k = min(self.n_support, self.n_pseudo_per_way)
            for w_idx in range(self.n_way):
                per_class = support[w_idx]
                idx = torch.randperm(self.n_support)[:select_k]
                selected = per_class[idx]
                pseudo_list = []
                while len(pseudo_list) < self.n_pseudo_per_way:
                    for s in selected:
                        pseudo_list.append(random_aug(s))
                        if len(pseudo_list) >= self.n_pseudo_per_way:
                            break
                pseudo_all.append(torch.stack(pseudo_list))
            pseudo = torch.stack(pseudo_all)

        all_per_class = torch.cat([support, pseudo], dim=1)  # [n_way, n_support + n_pseudo_per_way, C, H, W]
        psedo_set = all_per_class.view(-1, c, h, w)
        return psedo_set
