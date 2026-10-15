import unittest

from core.behavior_engine import (
    CirculacaoTracker,
    GestureTracker,
    LocalizadorBancada,
    classificar_pulso,
    selecionar_reconhecimentos,
)


class GestureDetectionTests(unittest.TestCase):
    def test_fast_raise_followed_by_fast_sideways_motion_is_suspicious(self):
        samples = [
            (0.0, 0.0, 1.0, 0.9),
            (0.2, 0.0, 0.0, 0.9),
            (0.4, 0.7, 0.0, 0.9),
        ]

        result = classificar_pulso(samples, 0.4)

        self.assertTrue(result["suspeito"])
        self.assertEqual(result["motivo"], "subida rápida seguida de golpe")

    def test_sequence_is_detected_with_sparse_pose_samples(self):
        samples = [
            (0.0, 0.0, 1.0, 0.9),
            (0.4, 0.0, 0.0, 0.9),
            (0.8, 0.8, 0.0, 0.9),
        ]

        result = classificar_pulso(samples, 0.8)

        self.assertTrue(result["suspeito"])

    def test_sparse_vertical_raise_and_lower_without_side_swing_is_not_suspicious(self):
        samples = [
            (0.0, 0.0, 0.8, 0.9),
            (2.0, 0.0, -0.1, 0.9),
            (4.0, 0.0, 0.8, 0.9),
        ]

        result = classificar_pulso(samples, 4.0, ciclo_s=2.0)

        self.assertFalse(result["suspeito"])

    def test_sequence_is_detected_when_pose_cycle_is_two_seconds(self):
        samples = [
            (0.0, 0.0, 0.8, 0.9),
            (2.0, 0.3, -0.1, 0.9),
            (4.0, 0.0, 0.8, 0.9),
            (6.0, 0.3, -0.1, 0.9),
            (8.0, 0.0, 0.8, 0.9),
        ]

        result = classificar_pulso(samples, 8.0, ciclo_s=2.0)

        self.assertTrue(result["suspeito"])
        self.assertEqual(result["motivo"], "braço indo e voltando bruscamente (amostragem lenta)")

    def test_sparse_raise_then_side_throw_does_not_require_unobservable_speed(self):
        samples = [
            (0.0, 0.0, 0.2, 0.9),
            (2.0, 0.0, -0.35, 0.9),
            (4.0, 0.35, -0.2, 0.9),
        ]

        result = classificar_pulso(samples, 4.0, ciclo_s=2.0)

        self.assertTrue(result["suspeito"])
        self.assertEqual(result["motivo"], "subida rápida seguida de golpe")

    def test_sparse_raise_with_diagonal_swing_is_suspicious(self):
        samples = [
            (0.0, 0.0, 0.2, 0.9),
            (2.0, 0.0, -0.35, 0.9),
            (4.0, 0.4, 0.0, 0.9),
        ]

        result = classificar_pulso(samples, 4.0, ciclo_s=2.0)

        self.assertTrue(result["suspeito"])

    def test_sparse_raise_then_forward_extension_is_suspicious(self):
        samples = [
            (0.0, 0.0, 1.0, 0.9, 0.35),
            (2.0, 0.0, 0.0, 0.9, 0.40),
            (4.0, 0.0, 0.0, 0.9, 0.82),
        ]

        result = classificar_pulso(samples, 4.0, ciclo_s=2.0)

        self.assertTrue(result["suspeito"])
        self.assertEqual(result["motivo"], "subida rápida seguida de golpe")

    def test_slow_raise_and_lower_with_sparse_samples_is_not_suspicious(self):
        samples = [
            (0.0, 0.0, 1.0, 0.9),
            (2.0, 0.0, 0.0, 0.9),
            (4.0, 0.0, 1.0, 0.9),
            (6.0, 0.0, 1.0, 0.9),
        ]

        result = classificar_pulso(samples, 6.0, ciclo_s=2.0)

        self.assertFalse(result["suspeito"])

    def test_alternating_sparse_arm_motion_is_suspicious(self):
        samples = [
            (0.0, 0.0, 0.8, 0.9),
            (0.5, 0.4, -0.1, 0.9),
            (1.0, 0.0, 0.8, 0.9),
        ]

        result = classificar_pulso(samples, 1.0, ciclo_s=0.5)

        self.assertTrue(result["suspeito"])
        self.assertEqual(result["motivo"], "braço indo e voltando bruscamente (amostragem lenta)")

    def test_very_slow_sampling_detects_two_alternating_jumps(self):
        samples = [
            (0.0, 0.0, 0.8, 0.9),
            (1.3, 0.4, -0.1, 0.9),
            (2.6, 0.0, 0.8, 0.9),
        ]

        result = classificar_pulso(samples, 2.6, ciclo_s=1.3)

        self.assertTrue(result["suspeito"])

    def test_moderate_side_to_side_swing_is_detected_with_slow_sampling(self):
        samples = [
            (0.0, -0.25, -0.1, 0.9),
            (1.6, 0.25, -0.1, 0.9),
            (3.2, -0.25, -0.1, 0.9),
        ]

        result = classificar_pulso(samples, 3.2, ciclo_s=1.6)

        self.assertTrue(result["suspeito"])
        self.assertEqual(result["motivo"], "braço indo e voltando bruscamente (amostragem lenta)")

    def test_action_sensitivity_changes_minimum_swing_amplitude(self):
        samples = [
            (0.0, -0.25, -0.1, 0.9),
            (1.6, 0.25, -0.1, 0.9),
            (3.2, -0.25, -0.1, 0.9),
        ]

        self.assertFalse(classificar_pulso(samples, 3.2, ciclo_s=1.6, sensibilidade=1)["suspeito"])
        self.assertTrue(classificar_pulso(samples, 3.2, ciclo_s=1.6, sensibilidade=10)["suspeito"])

    def test_slow_raise_followed_by_side_to_side_swing_is_suspicious(self):
        samples = [
            (0.0, 0.0, 0.8, 0.9),
            (1.0, 0.0, 0.2, 0.9),
            (2.0, 0.0, -0.1, 0.9),
            (3.0, 0.25, -0.1, 0.9),
            (4.0, -0.25, -0.1, 0.9),
            (5.0, 0.25, -0.1, 0.9),
        ]

        result = classificar_pulso(samples, 5.0, ciclo_s=1.0, sensibilidade=10)

        self.assertTrue(result["suspeito"])

    def test_fast_vertical_raise_and_lower_alone_is_not_suspicious(self):
        samples = [
            (0.0, 0.0, 1.0, 0.9),
            (0.2, 0.0, 0.0, 0.9),
            (0.4, 0.0, 1.0, 0.9),
            (0.6, 0.0, 1.0, 0.9),
        ]

        result = classificar_pulso(samples, 0.6, ciclo_s=0.2)

        self.assertFalse(result["suspeito"])

    def test_sparse_slow_raise_and_hold_without_swing_is_not_suspicious(self):
        samples = [
            (0.0, 0.0, 1.0, 0.9),
            (2.0, 0.0, 0.0, 0.9),
            (4.0, 0.0, 0.0, 0.9),
            (6.0, 0.0, 0.0, 0.9),
        ]

        result = classificar_pulso(samples, 6.0, ciclo_s=2.0)

        self.assertFalse(result["suspeito"])

    def test_slow_raise_and_held_arm_is_not_suspicious(self):
        samples = [
            (0.0, 0.0, 1.0, 0.9),
            (0.25, 0.0, 0.7, 0.9),
            (0.5, 0.0, 0.4, 0.9),
            (0.75, 0.0, 0.1, 0.9),
            (1.0, 0.0, -0.1, 0.9),
            (1.25, 0.0, -0.1, 0.9),
            (1.5, 0.0, -0.1, 0.9),
        ]

        result = classificar_pulso(samples, 1.5)

        self.assertFalse(result["suspeito"])

    def test_fast_raise_and_hold_without_second_motion_is_not_suspicious(self):
        samples = [
            (0.0, 0.0, 1.0, 0.9),
            (0.2, 0.0, 0.0, 0.9),
            (0.4, 0.0, 0.0, 0.9),
            (0.6, 0.0, 0.0, 0.9),
        ]

        result = classificar_pulso(samples, 0.6)

        self.assertFalse(result["suspeito"])

    def test_tracker_reports_temporal_event_once(self):
        tracker = GestureTracker()
        shoulders = ((100.0, 100.0, 0.9), (200.0, 100.0, 0.9))
        frames = [
            (0.0, (150.0, 200.0, 0.9)),
            (0.2, (150.0, 100.0, 0.9)),
            (0.4, (220.0, 100.0, 0.9)),
            (0.6, (220.0, 100.0, 0.9)),
        ]
        detected = []
        for timestamp, left_wrist in frames:
            person = {
                "shoulders": shoulders,
                "wrists": (left_wrist, (0.0, 0.0, 0.0)),
            }
            suspicious, _ = tracker.update(timestamp, [person], ciclo_s=0.2)
            detected.append(suspicious)

        self.assertEqual(detected, [False, False, True, False])

    def test_outside_stationary_student_is_not_marked_as_circulating(self):
        tracker = CirculacaoTracker()

        self.assertFalse(tracker.update("aluno-1", True, 100, 100, 40, 0.0))
        self.assertFalse(tracker.update("aluno-1", True, 102, 100, 40, 0.4))
        self.assertFalse(tracker.update("aluno-1", True, 101, 99, 40, 0.8))

    def test_outside_movement_is_marked_as_circulating_after_confirmation(self):
        tracker = CirculacaoTracker()

        self.assertFalse(tracker.update("aluno-1", True, 100, 100, 40, 0.0))
        self.assertFalse(tracker.update("aluno-1", True, 110, 100, 40, 0.3))
        self.assertTrue(tracker.update("aluno-1", True, 135, 100, 40, 0.7))

    def test_circulation_history_survives_slow_face_inference(self):
        tracker = CirculacaoTracker()

        self.assertFalse(tracker.update("aluno-1", True, 100, 100, 40, 0.0))
        self.assertFalse(tracker.update("aluno-1", True, 120, 100, 40, 1.8))
        self.assertTrue(tracker.update("aluno-1", True, 145, 100, 40, 3.6))

    def test_single_position_jump_is_not_enough_to_mark_circulating(self):
        tracker = CirculacaoTracker()

        self.assertFalse(tracker.update("aluno-1", True, 100, 100, 40, 0.0))
        self.assertFalse(tracker.update("aluno-1", True, 140, 100, 40, 0.3))
        self.assertFalse(tracker.update("aluno-1", True, 140, 100, 40, 0.7))

    def test_returning_to_a_bench_resets_outside_movement(self):
        tracker = CirculacaoTracker()
        tracker.update("aluno-1", True, 100, 100, 40, 0.0)
        self.assertFalse(tracker.update("aluno-1", False, 140, 100, 40, 0.3))
        self.assertFalse(tracker.update("aluno-1", True, 140, 100, 40, 0.4))

    def test_clear_bench_change_is_immediate(self):
        locator = LocalizadorBancada([
            {"nome": "A", "pontos": [(0, 0), (100, 0), (100, 100), (0, 100)]},
            {"nome": "B", "pontos": [(105, 0), (205, 0), (205, 100), (105, 100)]},
        ])
        state = {}

        self.assertEqual(locator.localizar(state, 50, 50, 20, 0.0), "A")
        self.assertEqual(locator.localizar(state, 115, 50, 20, 0.1), "B")

    def test_transition_between_nearby_benches_does_not_show_free_area(self):
        locator = LocalizadorBancada([
            {"nome": "A", "pontos": [(0, 0), (100, 0), (100, 100), (0, 100)]},
            {"nome": "B", "pontos": [(105, 0), (205, 0), (205, 100), (105, 100)]},
        ])
        state = {}

        self.assertEqual(locator.localizar(state, 50, 50, 20, 0.0), "A")
        self.assertEqual(locator.localizar(state, 102, 50, 20, 1.0), "A")
        self.assertEqual(locator.localizar(state, 104, 50, 20, 1.1), "B")
        self.assertEqual(locator.localizar(state, 102, 50, 20, 1.2), "B")

    def test_leaving_bench_follows_on_second_outside_sample(self):
        locator = LocalizadorBancada([
            {"nome": "A", "pontos": [(0, 0), (100, 0), (100, 100), (0, 100)]},
        ])
        state = {}

        self.assertEqual(locator.localizar(state, 50, 50, 20, 0.0), "A")
        # logo na borda de fora, um único frame é tolerado (ruído do detector)...
        self.assertEqual(locator.localizar(state, 110, 50, 20, 0.5), "A")
        # ...mas o segundo frame seguido fora já vira Área Livre (sem esperar 2s)
        self.assertEqual(locator.localizar(state, 110, 50, 20, 0.6), "Área Livre")

    def test_clearly_outside_leaves_bench_in_the_same_frame(self):
        locator = LocalizadorBancada([
            {"nome": "A", "pontos": [(0, 0), (100, 0), (100, 100), (0, 100)]},
        ])
        state = {}

        self.assertEqual(locator.localizar(state, 50, 50, 20, 0.0), "A")
        self.assertEqual(locator.localizar(state, 150, 50, 20, 0.1), "Área Livre")

    def test_leaving_bench_with_slow_camera_does_not_wait_grace_time(self):
        locator = LocalizadorBancada([
            {"nome": "A", "pontos": [(0, 0), (100, 0), (100, 100), (0, 100)]},
        ])
        state = {}

        self.assertEqual(locator.localizar(state, 50, 50, 20, 0.0), "A")
        self.assertEqual(locator.localizar(state, 110, 50, 20, 1.0), "A")
        self.assertEqual(locator.localizar(state, 110, 50, 20, 1.5), "Área Livre")

    def test_single_noisy_frame_outside_returns_to_same_bench(self):
        locator = LocalizadorBancada([
            {"nome": "A", "pontos": [(0, 0), (100, 0), (100, 100), (0, 100)]},
        ])
        state = {}

        self.assertEqual(locator.localizar(state, 50, 50, 20, 0.0), "A")
        self.assertEqual(locator.localizar(state, 110, 50, 20, 0.1), "A")
        self.assertEqual(locator.localizar(state, 60, 50, 20, 0.2), "A")
        self.assertEqual(locator.localizar(state, 110, 50, 20, 0.3), "A")

    def test_reaching_another_bench_cancels_outside_grace_and_switches_directly(self):
        locator = LocalizadorBancada([
            {"nome": "A", "pontos": [(0, 0), (100, 0), (100, 100), (0, 100)]},
            {"nome": "B", "pontos": [(200, 0), (300, 0), (300, 100), (200, 100)]},
        ])
        state = {}

        self.assertEqual(locator.localizar(state, 50, 50, 20, 0.0), "A")
        self.assertEqual(locator.localizar(state, 110, 50, 20, 0.5), "A")
        self.assertEqual(locator.localizar(state, 250, 50, 20, 0.8), "B")
        self.assertNotIn("fora_desde", state)

    def test_single_outside_sample_does_not_insert_free_area_between_benches(self):
        locator = LocalizadorBancada([
            {"nome": "A", "pontos": [(0, 0), (100, 0), (100, 100), (0, 100)]},
            {"nome": "B", "pontos": [(200, 0), (300, 0), (300, 100), (200, 100)]},
        ])
        state = {}

        self.assertEqual(locator.localizar(state, 50, 50, 20, 0.0), "A")
        self.assertEqual(locator.localizar(state, 110, 50, 20, 0.5), "A")
        self.assertEqual(locator.localizar(state, 250, 50, 20, 4.5), "B")

    def test_student_far_from_all_benches_is_area_free(self):
        locator = LocalizadorBancada([
            {"nome": "A", "pontos": [(0, 0), (100, 0), (100, 100), (0, 100)]},
            {"nome": "B", "pontos": [(105, 0), (205, 0), (205, 100), (105, 100)]},
        ])
        state = {}

        self.assertEqual(locator.localizar(state, 50, 50, 20, 0.0), "A")
        self.assertEqual(locator.localizar(state, 300, 50, 20, 1.0), "Área Livre")

    def test_known_face_is_not_reencoded_every_slow_camera_cycle(self):
        tracks = {
            1: {"size": 80, "last_rec": 10.0, "matricula": "aluno-1"},
        }

        self.assertEqual(selecionar_reconhecimentos(tracks, {1}, 14.5, 40), [])
        self.assertEqual(selecionar_reconhecimentos(tracks, {1}, 25.1, 40), [1])


if __name__ == "__main__":
    unittest.main()
