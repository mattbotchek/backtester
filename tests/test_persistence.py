import tempfile
import unittest
from pathlib import Path

from ml.persistence import atomic_pickle_dump


class PersistenceTests(unittest.TestCase):
    def test_failed_pickle_does_not_replace_existing_model(self):
        class Unpicklable:
            def __reduce__(self):
                raise RuntimeError("serialization failed")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            atomic_pickle_dump({"version": 1}, path)
            original = path.read_bytes()

            with self.assertRaisesRegex(RuntimeError, "serialization failed"):
                atomic_pickle_dump(Unpicklable(), path)

            self.assertEqual(original, path.read_bytes())
            self.assertEqual([path], list(Path(directory).iterdir()))


if __name__ == "__main__":
    unittest.main()
