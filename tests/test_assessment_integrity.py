"""Enforce the assessment requirement to preserve the supplied files unchanged."""

import hashlib
import unittest
from pathlib import Path


class AssessmentIntegrityTests(unittest.TestCase):
    def test_supplied_tools_and_restaurant_data_are_unchanged(self):
        project_root = Path(__file__).resolve().parents[1]
        expected_hashes = {
            "tools.py": "799eca54845208d907a02892821dec9471088506e77371d0faef329fb3b814f3",
            "restaurants.json": "131149d3e3a6b0d550adaa96bc93687c87adae809eb4f2c9997877dcdb0e9991",
        }

        for filename, expected_hash in expected_hashes.items():
            with self.subTest(filename=filename):
                digest = hashlib.sha256((project_root / filename).read_bytes()).hexdigest()
                self.assertEqual(digest, expected_hash, f"The supplied {filename} was modified.")


if __name__ == "__main__":
    unittest.main()
