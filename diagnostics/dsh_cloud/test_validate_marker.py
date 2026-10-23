"""CPU-only persistence and unfinished-run regressions; no DSH/model services."""
import ctypes
import ctypes.util
import json
from pathlib import Path
import tempfile
import unittest

from validate_marker import decode_frames, inspect


def compress(value):
    z = ctypes.CDLL(ctypes.util.find_library('zstd'))
    z.ZSTD_compressBound.argtypes, z.ZSTD_compressBound.restype = [ctypes.c_size_t], ctypes.c_size_t
    z.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    z.ZSTD_compress.restype = ctypes.c_size_t
    capacity = z.ZSTD_compressBound(len(value))
    source, target = ctypes.create_string_buffer(value), ctypes.create_string_buffer(capacity)
    size = z.ZSTD_compress(target, capacity, source, len(value), 1)
    if size > capacity:
        raise RuntimeError('Test fixture compression failed')
    return target.raw[:size]


class MarkerEvidenceTests(unittest.TestCase):
    def test_every_concatenated_frame_is_decoded(self):
        raw = compress(b'{"type":"session"}\n') + compress(b'{"type":"tool/call"}\n{"type":"tool/result"}\n')
        rows, count = decode_frames(raw)
        self.assertEqual(count, 2)
        self.assertEqual([row['type'] for row in rows], ['session', 'tool/call', 'tool/result'])

    def test_partial_trailing_frame_is_not_success(self):
        raw = compress(b'{"type":"session"}\n') + compress(b'{"type":"tool/result"}\n')[:-2]
        with self.assertRaisesRegex(ValueError, 'Incomplete or invalid'):
            decode_frames(raw)

    def test_unfinished_case_is_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = inspect(root, root / 'bridge', root / 'tokenizer')
            self.assertEqual(result['status'], 'pending')
            self.assertFalse(result['passed'])

    def test_failed_exit_cannot_pass_without_tool_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'exit.json').write_text(json.dumps({'exit_code': 1}))
            result = inspect(root, root / 'bridge', root / 'tokenizer')
            self.assertEqual(result['status'], 'failed')
            self.assertIn('exited with code 1', result['reason'])


if __name__ == '__main__':
    unittest.main()
