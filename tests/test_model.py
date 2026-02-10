import unittest
import torch
from src.models.transformer_net import ChessTransformerNetV2

class TestChessTransformerNetV2(unittest.TestCase):
    def test_forward_pass(self):
        # Batch size 4, 116 planes, 8x8
        x = torch.randn(4, 116, 8, 8)
        model = ChessTransformerNetV2(num_input_planes=116)

        policy, value = model(x)

        # Policy: (B, 4096)
        self.assertEqual(policy.shape, (4, 4096))
        # Value: (B, 1)
        self.assertEqual(value.shape, (4, 1))

if __name__ == '__main__':
    unittest.main()
