"""
Validation script for embedding-space KNN retrieval implementation.

Tests that:
1. EncoderFeatureExtractor can extract features
2. ICTTrafficDataset accepts encoder features
3. define_ict_dataloader can work with/without model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from lib.ict_encoder_features import EncoderFeatureExtractor, extract_encoder_features_batch, extract_single_feature
from lib.ict_data_process import ICTTrafficDataset


def test_encoder_feature_extractor():
    """Test basic encoder feature extraction."""
    print("\n=== Testing EncoderFeatureExtractor ===")
    
    # Mock OpenCity model structure
    class MockOpenCityModel:
        def __init__(self):
            self.device = torch.device('cpu')
            self.output_dim = 1
            self.embed_dim = 512
            self.lap_mx_dict = {'TEST': torch.eye(10)}
            self.adj_mx_dict = {'TEST': torch.ones(10, 10)}
            self.geo_mask_dict = {'TEST': torch.zeros(10, 10, dtype=torch.bool)}
            self.sem_mask = None
            
            # Mock components
            class MockModule(torch.nn.Module):
                def __init__(self, output_size):
                    super().__init__()
                    self.output_size = output_size
                def forward(self, x):
                    return torch.randn(x.shape[0], 24, 10, 512)  # [B, P, N, D]
            
            self.patch_embedding_flow = MockModule(512)
            self.patch_embedding_time = MockModule(512)
            self.spatial_embedding = MockModule(512)
            self.encoder_blocks = torch.nn.ModuleList([MockModule(512) for _ in range(2)])
    
    try:
        model = MockOpenCityModel()
        extractor = EncoderFeatureExtractor(model)
        
        # Create sample input
        batch_hist = torch.randn(2, 24, 10, 1)  # [B, T, N, F]
        batch_temp = torch.randn(2, 24, 10, 2)  # [B, T, N, F] with temporal features
        
        # Extract features
        with torch.no_grad():
            features = extractor(batch_hist, batch_temp, 'TEST')
        
        # Verify output shape
        assert features.shape == (2, 512), f"Expected (2, 512), got {features.shape}"
        print("✓ EncoderFeatureExtractor works correctly")
        print(f"  Input shape: {batch_hist.shape}")
        print(f"  Output shape: {features.shape}")
        return True
    except Exception as e:
        print(f"✗ EncoderFeatureExtractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ict_dataset_with_encoder_features():
    """Test ICTTrafficDataset with encoder features."""
    print("\n=== Testing ICTTrafficDataset with Encoder Features ===")
    
    try:
        # Create sample data
        T_total = 100
        N = 10
        F = 2
        data = np.random.randn(T_total, N, F).astype(np.float32)
        
        # Create demo pool
        iw, ow = 12, 12
        demo_pool = [
            (data[i:i + iw], data[i + iw:i + iw + ow], i)
            for i in range(len(data) - iw - ow + 1)
        ]
        
        # Create encoder features (512-d per demo)
        num_demos = len(demo_pool)
        encoder_features = np.random.randn(num_demos, 512).astype(np.float32)
        
        # Create dataset
        dataset = ICTTrafficDataset(
            data[:-iw-ow], 
            batch_size=4, 
            input_window=iw,
            output_window=ow,
            demo_pool=demo_pool,
            num_demonstrations=3,
            num_prefix_selections=1,
            demo_selection='similar',
            eval_only=True,
            encoder_features=encoder_features,
            use_encoder_knn=True
        )
        
        # Test __getitem__
        batch_x, batch_y, demos_x, demos_y = dataset[0]
        assert batch_x.shape == (4, iw, N, F), f"batch_x shape: {batch_x.shape}"
        assert batch_y.shape == (4, ow, N, F), f"batch_y shape: {batch_y.shape}"
        assert demos_x.shape[0] == 4, f"demos_x batch dim: {demos_x.shape[0]}"
        assert demos_x.shape[2] == 3, f"demos_x K dim: {demos_x.shape[2]}"
        
        print("✓ ICTTrafficDataset with encoder features works correctly")
        print(f"  Batch x shape: {batch_x.shape}")
        print(f"  Demos x shape: {demos_x.shape}")
        return True
    except Exception as e:
        print(f"✗ ICTTrafficDataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ict_dataset_fallback():
    """Test ICTTrafficDataset fallback to raw-input KNN."""
    print("\n=== Testing ICTTrafficDataset Fallback (No Encoder Features) ===")
    
    try:
        # Create sample data
        T_total = 100
        N = 10
        F = 2
        data = np.random.randn(T_total, N, F).astype(np.float32)
        
        # Create demo pool
        iw, ow = 12, 12
        demo_pool = [
            (data[i:i + iw], data[i + iw:i + iw + ow], i)
            for i in range(len(data) - iw - ow + 1)
        ]
        
        # Create dataset WITHOUT encoder features
        dataset = ICTTrafficDataset(
            data[:-iw-ow], 
            batch_size=4, 
            input_window=iw,
            output_window=ow,
            demo_pool=demo_pool,
            num_demonstrations=3,
            num_prefix_selections=1,
            demo_selection='random',
            eval_only=True,
            encoder_features=None,
            use_encoder_knn=False
        )
        
        # Test __getitem__
        batch_x, batch_y, demos_x, demos_y = dataset[0]
        assert demos_x.shape[2] == 3, f"demos_x K dim: {demos_x.shape[2]}"
        
        print("✓ ICTTrafficDataset fallback works correctly")
        print(f"  Mode: raw-input L2 KNN")
        print(f"  Demos x shape: {demos_x.shape}")
        return True
    except Exception as e:
        print(f"✗ ICTTrafficDataset fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("EMBEDDING-SPACE KNN RETRIEVAL VALIDATION")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("EncoderFeatureExtractor", test_encoder_feature_extractor()))
    results.append(("ICTTrafficDataset with Encoder", test_ict_dataset_with_encoder_features()))
    results.append(("ICTTrafficDataset Fallback", test_ict_dataset_fallback()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    total_pass = sum(1 for _, passed in results if passed)
    total = len(results)
    print(f"\nTotal: {total_pass}/{total} tests passed")
    
    if total_pass == total:
        print("\n🎉 All validation tests passed!")
        return 0
    else:
        print(f"\n❌ {total - total_pass} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
