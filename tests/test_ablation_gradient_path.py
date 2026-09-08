"""CPU regression of actual attack methods, without CLIP downloads or training."""
import ast
from pathlib import Path
import sys
import unittest
import warnings
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.tool_func import calc_mean_std, fgsm_attack, changeNewAdvStyle
from scripts.run_source_only_ablation import ARMS, commands

SOURCE = (ROOT / 'methods/StyleAdv_RN_GNN.py').read_text(encoding='utf-8-sig')
CLASS = next(n for n in ast.parse(SOURCE).body if isinstance(n, ast.ClassDef) and n.name == 'StyleAdvGNN')
def actual_method(name):
    node = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(torch=torch, calc_mean_std=calc_mean_std, fgsm_attack=fgsm_attack,
                 changeNewAdvStyle=changeNewAdvStyle, warnings=warnings, is_main_process=lambda: False)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-method>', 'exec'), scope)
    return scope[name]

class Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(3, 3, 1), nn.Tanh()) for _ in range(4)])
    def forward_block1(self, x): return self.blocks[0](x)
    def forward_block2(self, x): return self.blocks[1](x)
    def forward_block3(self, x): return self.blocks[2](x)
    def forward_block4(self, x): return self.blocks[3](x)
    def forward_rest(self, x): return x.mean((2, 3))

class Toy(nn.Module):
    adversarial_attack_Incre = actual_method('adversarial_attack_Incre')
    _compute_text_guidance = actual_method('_compute_text_guidance')
    def __init__(self):
        super().__init__()
        self.feature = Features()
        self.classifier = nn.Linear(3, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.semantic_drift_control_enabled = False
        self.text_guide_epsilon, self.text_guide_gradient = True, False
        self.n_way, self.device = 2, torch.device('cpu')
        self.use_prompt_ensemble, self._warned_missing_prompts = False, True
        self.text_weight_mode = 'fixed'
        self.epsilon_scale_min, self.epsilon_scale_max = .3, 1.7
        self.style_prompt_generator = nn.Linear(4, 8)
        self.epsilon_scale_head = nn.Sequential(nn.Linear(8, 3), nn.Sigmoid())
        self.embedding = torch.randn(4, requires_grad=True)
        object.__setattr__(self, 'text_prompt_encoder', self)
    def encode_class_prompts(self, prompts): return None, self.embedding
    def _get_class_prompts(self, ids): return ['class'] * len(ids)
    def _resolve_text_guidance_weight(self, stats): return 1.

class Regression(unittest.TestCase):
    def test_first_order_scale_gradient_both_modes(self):
        torch.set_num_threads(1)
        for mode in ('independent', 'progressive'):
            torch.manual_seed(0)
            model = Toy()
            x = torch.randn(2, 2, 3, 4, 4)
            y = torch.tensor([[0, 0], [1, 1]])
            scales, _ = model._compute_text_guidance(y, 2)
            self.assertTrue(scales.requires_grad)
            stats = model.adversarial_attack_Incre(x, y, [.8, .08, .008], scales, attack_mode=mode)
            self.assertTrue(all(p.grad is None for p in model.feature.parameters()))
            h = x.flatten(0, 1)
            for i in range(3):
                h = model.feature.blocks[i](h)
                h = changeNewAdvStyle(h, stats[2*i], stats[2*i+1], 0)
            loss = model.loss_fn(model.classifier(model.feature.forward_rest(model.feature.forward_block4(h))), y.flatten())
            loss.backward()
            grad = model.epsilon_scale_head[0].weight.grad
            self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum() > 0)
            self.assertIsNone(model.embedding.grad)
            before = model.epsilon_scale_head[0].weight.detach().clone()
            torch.optim.SGD(model.parameters(), lr=.1).step()
            self.assertFalse(torch.equal(before, model.epsilon_scale_head[0].weight))

    def test_runner_cli_compatibility(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from options import parse_args
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(data_dir=Path(directory), smoke=True)
            for arm in ARMS:
                for kind, command in zip(('train', 'test'), commands(args, arm, 5, 'cli_check')):
                    command[command.index('--save_dir') + 1] = directory
                    with patch.object(sys, 'argv', command[1:]):
                        parsed = parse_args(kind)
                    self.assertEqual(parsed.n_shot, 5)

    def test_runner_arms(self):
        from types import SimpleNamespace
        args = SimpleNamespace(data_dir=Path('/data'), smoke=False)
        for arm in ARMS:
            train, test = commands(args, arm, 5, 'test')
            self.assertEqual(train[train.index('--target_ssl_weight')+1], '0')
            self.assertNotIn('--target_dataset', train)
            self.assertEqual('--use_text_guidance' in train, arm in 'DE')
            self.assertEqual('--disable_style_adv_generator' in train, arm == 'A')
            self.assertEqual(test[test.index('--n_episodes_test')+1], '1000')
            self.assertEqual(test[test.index('--eval_seed')+1], '0')

if __name__ == '__main__':
    unittest.main()
