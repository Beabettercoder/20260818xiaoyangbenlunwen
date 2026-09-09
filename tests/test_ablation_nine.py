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


if __name__ == '__main__':
    unittest.main()
