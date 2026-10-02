import numpy as np
import pytest

from grasp_failure_prediction.integrations.hug_rng import seed_cpu_generation


@pytest.mark.parametrize('seed', [-1, 2**32, True, 1.5])
def test_invalid_cpu_seed_is_rejected_before_generators_are_touched(seed):
    class Untouched:
        def manual_seed(self, value):
            pytest.fail('Invalid seed reached PyTorch')
    with pytest.raises(ValueError, match='unsigned 32-bit'):
        seed_cpu_generation(seed, torch_module=Untouched())


def test_cpu_fps_and_flow_noise_repeat_after_seeding():
    torch = pytest.importorskip('torch')
    cluster = pytest.importorskip('torch_cluster')
    points = torch.from_numpy(np.random.default_rng(8).normal(size=(100, 3)).astype(np.float32))

    def draw(seed):
        seed_cpu_generation(seed, torch_module=torch)
        return cluster.fps(points, ratio=.1, random_start=True), torch.randn(99)

    first_fps, first_noise = draw(42)
    # An intervening request must not affect a repeat of an earlier seed.
    draw(17)
    repeated_fps, repeated_noise = draw(42)
    assert torch.equal(first_fps, repeated_fps)
    assert torch.equal(first_noise, repeated_noise)
