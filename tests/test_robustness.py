import numpy as np
from ivafr.preprocess.degrade2d import occlude


def test_sunglasses_severity_and_input_immutability():
    image = np.arange(10000, dtype=np.float32).reshape(100, 100)
    original = image.copy()
    for fraction in (0.1, 0.2, 0.4):
        result = occlude(image, "sunglasses", fraction)
        assert abs(np.mean(result != image) - fraction) < 0.01
    assert np.array_equal(image, original)


def test_occlusion_reproducibility():
    image = np.arange(4096).reshape(64, 64)
    assert np.array_equal(occlude(image, "block", 0.2, 7), occlude(image, "block", 0.2, 7))
