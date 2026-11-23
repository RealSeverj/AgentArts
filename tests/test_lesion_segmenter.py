import unittest

import numpy as np

from inference.lesion_segmenter import _restore

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires optional inference torch runtime")
class LesionCoordinateTests(unittest.TestCase):
    def test_padding_excluded_and_native_positions_preserved(self):
        grid = torch.zeros(5, 5, 6)
        grid[0].fill_(1)
        grid[3, 0, :].fill_(2)  # A foreground class in the padded area.
        grid[2, 1, 1] = 3
        grid[4, 2, 3] = 3
        masks = _restore(grid, (8, 4), (3, 1, 6, 3), (1, 1, 3, 2))
        self.assertEqual({key: mask.size for key, mask in masks.items()},
                         dict.fromkeys(("SE", "EX", "HE", "MA"), (8, 4)))
        for code, location in (("EX", (1, 3)), ("MA", (2, 5))):
            expected = np.zeros((4, 8), dtype=np.uint8)
            expected[location] = 255
            np.testing.assert_array_equal(np.asarray(masks[code]), expected)
        self.assertIsNone(masks["HE"].getbbox())
        self.assertIsNone(masks["SE"].getbbox())

    def test_all_background_and_ties_return_empty_masks_at_odd_resolution(self):
        grid = torch.ones(5, 2, 3)
        masks = _restore(grid, (11, 7), (1, 1, 9, 6), (0, 0, 3, 2))
        self.assertTrue(all(mask.mode == "L" and mask.size == (11, 7) and
                            mask.getbbox() is None for mask in masks.values()))
