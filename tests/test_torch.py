import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tsclahe import TissueSNRCLAHE
from tsclahe.torch_backend import (TileController, prepare_tensors, forward_prepared,
                                   TorchPredictor, clipped_luts_torch, save_checkpoint,
                                   load_predictor)
from tsclahe.core import clipped_luts


def test_torch_numpy_lut_parity():
    hist = np.random.default_rng(8).dirichlet(np.full(64, .2), size=(3, 4)).astype(np.float32)
    clips = np.full((3, 4), 2.3, np.float32)
    got = clipped_luts_torch(torch.tensor(hist[None]), torch.tensor(clips[None]))[0].numpy()
    np.testing.assert_allclose(got, clipped_luts(hist, clips), atol=3e-7)


def test_clip_limit_gradcheck():
    torch.manual_seed(1)
    hist = torch.rand(1, 2, 3, 16, dtype=torch.float64) ** 4
    clips = torch.full((1, 2, 3), 2.31, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda c: clipped_luts_torch(hist, c), (clips,), eps=1e-6, atol=1e-5)


def test_learned_backend_parity_and_gradients(scene, config):
    image, mask = scene
    initial = TissueSNRCLAHE(config)(image, mask=mask, limits=(0, 1))
    model = TileController(config)
    tensors = prepare_tensors(initial.normalized, mask, initial.transform.statistics, config)
    output, clips, strength = forward_prepared(model, tensors, initial.transform.grid)
    clips.retain_grad()
    strength.retain_grad()
    output.square().mean().backward()
    assert clips.grad is not None and torch.isfinite(clips.grad).all() and clips.grad.abs().sum() > 0
    assert strength.grad is not None and strength.grad.abs().sum() > 0
    assert model.network[-1].weight.grad.abs().sum() > 0
    result = TissueSNRCLAHE(config, predictor=TorchPredictor(model))(image, mask=mask, limits=(0, 1))
    np.testing.assert_allclose(result.enhanced, output.detach().numpy()[0, 0], atol=5e-7)
    np.testing.assert_array_equal(result.enhanced[~mask], image[~mask])


def test_checkpoint_roundtrip(tmp_path, scene, config):
    image, mask = scene
    model = TileController(config)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, metadata={"kind": "test_only"})
    loaded = load_predictor(path)
    r1 = TissueSNRCLAHE(config, predictor=TorchPredictor(model))(image, mask=mask, limits=(0, 1))
    r2 = TissueSNRCLAHE(config, predictor=loaded)(image, mask=mask, limits=(0, 1))
    np.testing.assert_array_equal(r1.enhanced, r2.enhanced)
    assert loaded.metadata["kind"] == "test_only"
