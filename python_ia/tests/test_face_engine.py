import unittest
from unittest.mock import patch

from core.face_engine import FaceEngine


class FaceEngineMatchTests(unittest.TestCase):
    def setUp(self):
        self.engine = FaceEngine.__new__(FaceEngine)
        self.engine.known_encodings = ["aluno-1", "aluno-2"]
        self.engine.known_names = ["aluno-1", "aluno-2"]

    def test_clear_match_is_immediately_confirmed(self):
        with patch("core.face_engine.face_recognition.face_distance", return_value=[0.32, 0.51]):
            details = self.engine.find_best_match_details("amostra")
            identity = self.engine.find_best_match("amostra")

        self.assertEqual(details[0], "aluno-1")
        self.assertAlmostEqual(details[1], 0.32)
        self.assertAlmostEqual(details[2], 0.19)
        self.assertTrue(details[3])
        self.assertEqual(identity, "aluno-1")

    def test_valid_but_less_clear_match_requires_temporal_confirmation(self):
        with patch("core.face_engine.face_recognition.face_distance", return_value=[0.46, 0.54]):
            details = self.engine.find_best_match_details("amostra")

        self.assertEqual(details[0], "aluno-1")
        self.assertFalse(details[3])

    def test_ambiguous_match_remains_unknown(self):
        with patch("core.face_engine.face_recognition.face_distance", return_value=[0.40, 0.42]):
            details = self.engine.find_best_match_details("amostra")

        self.assertEqual(details[0], "Desconhecido")
        self.assertFalse(details[3])

    def test_invalid_distance_never_produces_an_identity(self):
        with patch("core.face_engine.face_recognition.face_distance", return_value=[float("nan"), 0.52]):
            details = self.engine.find_best_match_details("amostra")

        self.assertEqual(details[0], "Desconhecido")
        self.assertFalse(details[3])

    def test_multiple_references_for_same_student_do_not_reduce_identity_margin(self):
        self.engine.known_encodings = ["aluno-1-a", "aluno-1-b", "aluno-2"]
        self.engine.known_names = ["aluno-1", "aluno-1", "aluno-2"]

        with patch("core.face_engine.face_recognition.face_distance", return_value=[0.31, 0.33, 0.52]):
            details = self.engine.find_best_match_details("amostra")

        self.assertEqual(details[0], "aluno-1")
        self.assertAlmostEqual(details[2], 0.21)
        self.assertTrue(details[3])


if __name__ == "__main__":
    unittest.main()
