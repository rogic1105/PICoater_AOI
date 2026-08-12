import importlib.util
import io
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "backfill_hessian_standard_maps.py"
SPEC = importlib.util.spec_from_file_location("backfill_hessian_standard_maps", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BackfillHessianStandardMapsTests(unittest.TestCase):
    def test_hsm_round_trip_matches_product_format(self):
        encoded = MODULE.encode_hsm(np.asarray([[0.0, 0.25], [0.5, 1.0]], dtype=np.float16))
        self.assertEqual((2, 2), MODULE.validate_hsm(encoded))

    def test_archive_append_preserves_original_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "260101-010101.acap"
            grab = b"260101-010101"
            path.write_bytes(MODULE.FILE_MAGIC + struct.pack("<iqi", 1, 0, len(grab)) + grab)
            with path.open("ab") as writer:
                MODULE.append_record(writer, MODULE.Record(1, 1, 123, "frame-1", b"jpeg"))
            before = path.read_bytes()
            with path.open("ab") as writer:
                MODULE.append_record(writer, MODULE.Record(
                    14, 1, 123, "frame-1",
                    MODULE.encode_hsm(np.asarray([[0.5]], dtype=np.float16))))
            archive = MODULE.read_archive(path)
            self.assertTrue(path.read_bytes().startswith(before))
            self.assertEqual([1, 14], [record.kind for record in archive.records])
            self.assertEqual((1, 1), MODULE.validate_hsm(archive.records[1].payload))

    def test_archive_rebuild_replaces_maps_and_preserves_other_records(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.acap"
            target = Path(temp) / "target.acap"
            grab = b"260101-010101"
            source.write_bytes(MODULE.FILE_MAGIC + struct.pack("<iqi", 1, 0, len(grab)) + grab)
            raw = MODULE.Record(1, 1, 123, "frame-1", b"jpeg")
            curve = MODULE.Record(6, 1, 123, "frame-1", b"curve")
            old_map = MODULE.Record(14, 1, 123, "frame-1", MODULE.encode_hsm(
                np.zeros((5, 5), dtype=np.float16)))
            new_column = MODULE.Record(14, 1, 123, "frame-1", MODULE.encode_hsm(
                np.zeros((1, 1), dtype=np.float16)))
            new_row = MODULE.Record(15, 1, 123, "frame-1", MODULE.encode_hsm(
                np.ones((1, 1), dtype=np.float16)))
            with source.open("ab") as writer:
                for record in (raw, curve, old_map):
                    MODULE.append_record(writer, record)

            archive = MODULE.read_archive(source)
            retained = [record for record in archive.records
                        if record.kind not in (MODULE.HESSIAN_C, MODULE.HESSIAN_R)]
            MODULE.write_archive(target, archive, retained + [new_column, new_row])

            rebuilt = MODULE.read_archive(target)
            self.assertEqual([1, 6, 14, 15], [record.kind for record in rebuilt.records])
            self.assertEqual(b"jpeg", rebuilt.records[0].payload)
            self.assertEqual(b"curve", rebuilt.records[1].payload)
            self.assertEqual((1, 1), MODULE.validate_hsm(rebuilt.records[2].payload))

    def test_progress_pipe_failure_is_nonfatal(self):
        class BrokenWriter(io.StringIO):
            def write(self, value):
                raise OSError(22, "closed progress pipe")

        original = sys.stdout
        try:
            sys.stdout = BrokenWriter()
            MODULE.write_progress("progress")
        finally:
            sys.stdout = original


if __name__ == "__main__":
    unittest.main()
