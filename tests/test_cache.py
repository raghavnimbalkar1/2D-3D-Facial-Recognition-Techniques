import os
import numpy as np
import pytest
from ivafr.preprocess.cache import file_digest, is_cached, mark_cached


def test_content_change_with_same_size_and_timestamp_invalidates(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"abcd")
    stamp = path.stat()
    before = file_digest(path)
    path.write_bytes(b"abce")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert before != file_digest(path)


def test_cache_rejects_corrupted_or_absent_payload(tmp_path):
    path = tmp_path / "crop.npy"
    with pytest.raises(ValueError):
        mark_cached(path, {}, "input")
    np.save(path, np.ones((3, 3)))
    mark_cached(path, {}, "input")
    assert is_cached(path, {}, "input")
    path.write_bytes(b"corrupted")
    assert not is_cached(path, {}, "input")
