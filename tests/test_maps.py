"""지도 줄이기(cubecoach.segnet.maps_to): 앱이 256 입력의 64x64 지도를 48x48로 줄일 때 쓰는 계산."""

import numpy as np

from cubecoach.segnet import area_weights, maps_to


def test_area_weights_cover_each_input_cell_once():
    w = area_weights(64, 48)
    assert np.allclose(w.sum(0), 1.0)  # 입력 칸마다 넓이 1을 나눠 준다
    assert np.allclose(w.sum(1), 64 / 48)


def test_maps_to_keeps_probabilities_and_mean_heat():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(8), size=(64, 64)).transpose(2, 0, 1)
    heat = rng.uniform(0, 1, (64, 64)).astype(np.float32)
    lp, h = maps_to(np.log(p).astype(np.float32), heat, 48)
    assert lp.shape == (8, 48, 48) and h.shape == (48, 48)
    assert np.allclose(np.exp(lp).sum(0), 1.0, atol=1e-5)
    assert abs(float(h.mean()) - float(heat.mean())) < 1e-4


def test_maps_to_same_size_is_identity():
    lp = np.log(np.full((8, 48, 48), 1 / 8, np.float32))
    heat = np.zeros((48, 48), np.float32)
    a, b = maps_to(lp, heat, 48)
    assert a is lp and b is heat
