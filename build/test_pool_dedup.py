import unittest
import numpy as np

def deduplicate_coconut_arrays(raw_keys, smis, mass, fps):
    k14_all = np.array([str(k)[:14] for k in raw_keys], dtype=object)
    _, first_idx = np.unique(k14_all, return_index=True)
    first_idx = np.sort(first_idx)
    return k14_all[first_idx], smis[first_idx], mass[first_idx], fps[first_idx]

class TestPoolDedup(unittest.TestCase):
    def test_coconut_dedup_synchronization(self):
        # In real InChIKeys, the first block is 14 uppercase letters.
        # Two stereoisomers of AAAAAAAAAAAAAA, one BBBBBBBBBBBBBB, one CCCCCCCCCCCCCC
        raw_keys = np.array([
            'AAAAAAAAAAAAAA-1234567890-N',
            'AAAAAAAAAAAAAA-0987654321-N',
            'BBBBBBBBBBBBBB-1111111111-N',
            'CCCCCCCCCCCCCC-2222222222-N'
        ], dtype=object)
        smis = np.array(['SMI_A1', 'SMI_A2', 'SMI_B1', 'SMI_C1'], dtype=object)
        mass = np.array([100.1, 100.1, 200.2, 300.3], dtype=np.float64)
        fps  = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.uint8)

        k14, u_smis, u_mass, u_fps = deduplicate_coconut_arrays(raw_keys, smis, mass, fps)

        self.assertEqual(len(k14), 3)
        self.assertEqual(list(k14), ['AAAAAAAAAAAAAA', 'BBBBBBBBBBBBBB', 'CCCCCCCCCCCCCC'])
        self.assertEqual(list(u_smis), ['SMI_A1', 'SMI_B1', 'SMI_C1'])
        self.assertEqual(list(u_mass), [100.1, 200.2, 300.3])
        self.assertEqual(u_fps.shape, (3, 2))
        np.testing.assert_array_equal(u_fps[0], [1, 0])

    def test_submission_dedup_logic(self):
        # Two stereoisomers sharing the same 14-char first block
        keys = [
            'AAAAAAAAAAAAAA-1234567890-N',
            'AAAAAAAAAAAAAA-0987654321-N',
            'BBBBBBBBBBBBBB-1111111111-N',
            'CCCCCCCCCCCCCC-2222222222-N'
        ]
        order = [0, 1, 2, 3]
        smis = ['SMI_A1', 'SMI_A2', 'SMI_B', 'SMI_C']

        seen = set()
        out_smis = []
        for idx in order:
            raw_k = keys[idx]
            if raw_k is None:
                continue
            k14 = str(raw_k)[:14]
            if len(k14) == 14 and k14 not in seen:
                seen.add(k14)
                out_smis.append(smis[idx])

        self.assertEqual(len(out_smis), 3)
        self.assertEqual(out_smis, ['SMI_A1', 'SMI_B', 'SMI_C'])

    def test_train_filtering_against_coconut(self):
        # COCONUT has AAAAAAAAAAAAAA, BBBBBBBBBBBBBB
        co_k14 = np.array(['AAAAAAAAAAAAAA', 'BBBBBBBBBBBBBB'], dtype=object)
        coco_set = set(co_k14)
        
        # Train has AAAAAAAAAAAAAA (duplicate with COCONUT) and DDDDDDDDDDDDDD (novel)
        tr_k14 = np.array(['AAAAAAAAAAAAAA', 'DDDDDDDDDDDDDD'], dtype=object)
        tr_mask = np.array([k not in coco_set for k in tr_k14], bool)
        
        filtered_tr = tr_k14[tr_mask]
        self.assertEqual(len(filtered_tr), 1)
        self.assertEqual(list(filtered_tr), ['DDDDDDDDDDDDDD'])

if __name__ == '__main__':
    unittest.main()
