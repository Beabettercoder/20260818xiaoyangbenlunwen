import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_ablation_nine import plans_for, SOURCES


class NineDirections(unittest.TestCase):
    def test_abcd_schedule(self):
        args = SimpleNamespace(sources=SOURCES, shots=[5, 1], arms=list('ABCD'),
                               tag='test', smoke=False)
        config = {s: dict(data_dir='/data/' + s, checkpoint='/weights/' + s + '/399.tar') for s in SOURCES}
        plans = plans_for(config, args)
        self.assertEqual(len(plans), 24)
        self.assertEqual(sum(p['train'] is not None for p in plans), 18)
        self.assertEqual(sum(len(p['targets']) for p in plans), 72)
        for p in plans:
            self.assertNotIn(p['source'], p['targets'])
            self.assertNotEqual(p['arm'], 'E')
            if p['arm'] == 'A':
                self.assertIsNone(p['train'])
                continue
            train = p['train']
            self.assertEqual(train[train.index('--source_dataset') + 1], p['source'])
            self.assertNotIn('--target_dataset', train)
            self.assertNotIn('--target_datasets', train)
            self.assertIn(p['source'], train[train.index('--warmup') + 1])

    def test_only_module_switches_differ_in_bcd(self):
        args = SimpleNamespace(sources=['NWPU'], shots=[5], arms=list('BCD'), tag='test', smoke=False)
        config = {'NWPU': dict(data_dir='/data', checkpoint='/weights/399.tar')}
        plans = plans_for(config, args)
        normalized = []
        for p in plans:
            cmd = list(p['train'])
            for key in ('--name', '--style_attack_mode', '--text_guide_epsilon'):
                i = cmd.index(key)
                del cmd[i:i+2]
            if '--use_text_guidance' in cmd:
                cmd.remove('--use_text_guidance')
            normalized.append(cmd)
            self.assertEqual(p['modules']['extra_epochs'], 200)
            self.assertNotIn('--disable_style_adv_generator', cmd)
            for key, value in {'--model': 'ResNet10', '--train_n_query': '5',
                               '--n_query': '15', '--stop_epoch': '200',
                               '--target_ssl_weight': '0', '--train_episodes': '100',
                               '--n_shot': '5'}.items():
                self.assertEqual(cmd[cmd.index(key)+1], value)
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(normalized[0], normalized[2])
        self.assertEqual([p['modules']['progressive'] for p in plans], [False, True, False])
        self.assertEqual([p['modules']['text_scales'] for p in plans], [False, False, True])

    def test_e_rejected(self):
        args = SimpleNamespace(sources=['NWPU'], shots=[5], arms=['E'], tag='test', smoke=False)
        with self.assertRaises(ValueError):
            plans_for({'NWPU': {}}, args)


if __name__ == '__main__':
    unittest.main()
