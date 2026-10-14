"""E10 functional tests; no scientific training or held-out data access."""
from pathlib import Path
import sys
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.sspg_spatial_mixer_matched_v2 import SpatialTokenMixer, SSPGSpatialMixerMatchedV2
from training.e53r_training_protocol_v1 import compute_joint_loss, mask_to_patch_target


class MixerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_local_interaction_without_far_or_batch_dependency(self):
        torch.manual_seed(101)
        mixer=SpatialTokenMixer().eval()
        x=torch.randn(2,196,256)
        modified=x.clone()
        modified[0,7*14+7,0]+=3
        with torch.no_grad():
            original=mixer(x)
            changed=mixer(modified)
        difference=(changed-original).abs().amax(dim=-1).reshape(2,14,14)
        self.assertGreater(float(difference[0,7,8]),1e-7)
        self.assertGreater(float(difference[0,7,9]),1e-9)
        far=torch.ones((14,14),dtype=torch.bool)
        far[5:10,5:10]=False
        self.assertEqual(float(difference[0][far].max()),0)
        self.assertEqual(float(difference[1].max()),0)
        x.requires_grad_(True)
        mixer(x)[0,7*14+8,0].backward()
        self.assertGreater(float(x.grad[0,7*14+7].abs().max()),0)
        self.assertEqual(float(x.grad[0,0].abs().max()),0)

    def test_parameters_are_active_and_shape_is_preserved(self):
        torch.manual_seed(102)
        mixer=SpatialTokenMixer()
        self.assertEqual(sum(p.numel() for p in mixer.parameters()),860288)
        x=torch.randn(2,196,256)
        output=mixer(x)
        self.assertEqual(tuple(output.shape),(2,196,256))
        output.square().mean().backward()
        for name,p in mixer.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(float(p.grad.abs().max()),0,name)
        with self.assertRaises(RuntimeError):
            mixer(torch.randn(1,195,256))

    def test_deterministic_initialization_and_eval(self):
        torch.manual_seed(103)
        one=SpatialTokenMixer().eval()
        torch.manual_seed(103)
        two=SpatialTokenMixer().eval()
        x=torch.randn(1,196,256)
        self.assertTrue(torch.equal(one(x),two(x)))

    def test_full_model_freezing_loss_and_gradient(self):
        self.assertTrue(torch.cuda.is_available())
        torch.manual_seed(104)
        model=SSPGSpatialMixerMatchedV2(str(ROOT/'checkpoints/clean_revision_v2/decoder_budget142/best.pth')).cuda()
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),1416100)
        self.assertIsNone(model.graph_frontend)
        self.assertFalse(model.uses_graph)
        self.assertTrue(model.uses_provenance)
        model.train()
        self.assertFalse(model.frozen_decoder().training)
        self.assertFalse(model.frozen_vit().training)
        self.assertTrue(model.frontend.fusion.training)
        self.assertFalse(any(p.requires_grad for p in model.frozen_decoder().parameters()))
        self.assertFalse(any(p.requires_grad for p in model.frozen_vit().parameters()))
        received=torch.rand(1,3,256,256,device='cuda')
        payload=torch.randint(0,2,(1,128),device='cuda').float()
        output=model(received,payload)
        self.assertEqual(tuple(output['head_tokens'].shape),(1,196,256))
        self.assertEqual(tuple(output['detection_logits'].shape),(1,))
        self.assertEqual(tuple(output['localization_grid_logits'].shape),(1,1,14,14))
        self.assertIsNone(output['graph_tokens'])
        loss=compute_joint_loss(output['detection_logits'],torch.ones(1,device='cuda'),
            output['localization_grid_logits'],mask_to_patch_target(torch.ones(1,1,256,256,device='cuda')))
        self.assertTrue(torch.isfinite(loss['total_loss']))
        loss['total_loss'].backward()
        for name,p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
            else:
                self.assertIsNone(p.grad,name)
        model.eval()
        with torch.no_grad():
            first=model(received,payload)
            second=model(received,payload)
        self.assertTrue(torch.equal(first['head_tokens'],second['head_tokens']))
        del model
        torch.cuda.empty_cache()


if __name__=='__main__':
    unittest.main(verbosity=2)
