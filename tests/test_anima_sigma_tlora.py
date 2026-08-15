from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "sd-scripts"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from library import config_util, flux_train_utils  # noqa: E402
from networks import tlora  # noqa: E402


class _FakeScheduler:
    class config:
        num_train_timesteps = 1000


class AnimaSigmaTloraSmokeTests(unittest.TestCase):
    def test_sigma_range_constrains_noise_sampling(self):
        sched = _FakeScheduler()
        args = types.SimpleNamespace(
            timestep_sampling="uniform",
            sigmoid_scale=1.0,
            discrete_flow_shift=3.0,
            ip_noise_gamma=None,
            weighting_scheme="uniform",
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=None,
        )
        latents = torch.randn(2, 4, 8, 8)
        noise = torch.randn(2, 4, 8, 8)
        cpu = torch.device("cpu")

        def sigmas(**kw):
            _, _, s = flux_train_utils.get_noisy_model_input_and_timesteps(
                args, sched, latents, noise, cpu, torch.float32, **kw
            )
            return s.flatten()

        s = sigmas(sigma_min=0.05, sigma_max=0.4)
        self.assertTrue(torch.all((s >= 0.05 - 1e-6) & (s <= 0.4 + 1e-6)))

        s = sigmas(sigma_min=0.4, sigma_max=0.4)
        self.assertTrue(torch.allclose(s, torch.full_like(s, 0.4)))

        s = sigmas(sigma_min=0.8, sigma_max=0.2)  # reversed -> swapped into [0.2, 0.8]
        self.assertTrue(torch.all((s >= 0.2 - 1e-6) & (s <= 0.8 + 1e-6)))

        s = sigmas()  # default path (no sigma)
        self.assertTrue(torch.all((s >= 0.0) & (s <= 1.0)))

    def test_config_injects_sigma_into_custom_attributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            Image.new("RGB", (64, 64), (200, 100, 50)).save(tmp / "a.png")
            (tmp / "a.txt").write_text("1girl, solo\n", encoding="utf-8")

            sub = config_util.SubsetBlueprint(
                params=config_util.DreamBoothSubsetParams(image_dir=str(tmp), num_repeats=1, caption_extension=".txt")
            )
            ds = config_util.DatasetBlueprint(
                is_dreambooth=True,
                is_controlnet=False,
                params=config_util.DreamBoothDatasetParams(
                    batch_size=1,
                    resolution=(64, 64),
                    enable_bucket=False,
                    sigma_min=0.35,
                    sigma_max=0.75,
                ),
                subsets=[sub],
            )
            bp = config_util.DatasetGroupBlueprint(datasets=[ds])
            train_group, _ = config_util.generate_dataset_group_by_blueprint(bp)

            self.assertEqual(len(train_group.datasets[0].subsets), 1)
            subset0 = train_group.datasets[0].subsets[0]
            self.assertEqual(subset0.custom_attributes.get("sigma_min"), 0.35)
            self.assertEqual(subset0.custom_attributes.get("sigma_max"), 0.75)

    def test_tlora_rank_mask_activates_and_clears(self):
        m = tlora.TLoRAModule(
            "lora_unet_block_0",
            torch.nn.Linear(4, 4),
            lora_dim=8,
            alpha=8,
            tlora_min_rank=2,
            tlora_rank_schedule="linear",
        )

        class FakeNet:
            current_timestep = None

        fake = FakeNet()
        m.set_network(fake)
        lx = torch.randn(1, 8)

        mask, scale = m._get_tlora_rank_mask_and_scale(lx)
        self.assertIsNone(mask)
        self.assertIsNone(scale)

        fake.current_timestep = torch.tensor([1.0])  # pure noise -> min rank
        mask, _ = m._get_tlora_rank_mask_and_scale(lx)
        self.assertIsNotNone(mask)
        self.assertEqual(int(mask.sum()), 2)

        fake.current_timestep = torch.tensor([0.0])  # clean -> full rank
        mask, _ = m._get_tlora_rank_mask_and_scale(lx)
        self.assertIsNotNone(mask)
        self.assertEqual(int(mask.sum()), 8)

        fake.current_timestep = None
        mask, scale = m._get_tlora_rank_mask_and_scale(lx)
        self.assertIsNone(mask)
        self.assertIsNone(scale)


if __name__ == "__main__":
    unittest.main()
