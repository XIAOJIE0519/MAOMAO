import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from maomao.data.event_extraction import observed_vasopressor_response, state_transitions
from maomao.data.clinical_events import OUTCOMES, extract_measurement_events
from maomao.data.event_sequence import EventSequenceDataset, TOKEN_KINDS, collate_event_sequences
from maomao.data.outcome_families import family_hit_counts, family_ids
from maomao.data.sampling import LengthSortedBatchSampler
from maomao.models.event_maomao import EventMAOMAO, event_time_loss
from maomao.evaluation.event_metrics import event_metric_report
from data.external_validation_common import MedicationResolver
from scripts.preprocess_event_sequences import (
    Record, _insert_clock_tokens, _insert_phase_summary_tokens,
)
from scripts.train import WarmupPlateauScheduler


def event_fixture(root: Path):
    meta = {
        "version": 2,
        "complete": True,
        "split": "all_train",
        "source_root": "/safe/train_data",
        "num_static": 2,
        "num_admissions": 1,
        "num_tokens": 6,
        "token_vocabulary": {"<PAD>": 0, "<BOS>": 1, "med:x": 2,
                             "event:a": 3, "event:b": 4, "event:stable": 5},
        "outcome_vocabulary": ["stable", "a", "b"],
        "audit_counts": {"stable": 5, "a": 20, "b": 10},
        "trajectory_horizons_hours": [0.1],
    }
    (root / "event_sequence_meta.json").write_text(json.dumps(meta))
    np.save(root / "sequence_ptr.npy", np.array([0, 6], dtype=np.int64))
    np.save(root / "static_baseline.npy", np.array([[0.5, 1.0]], dtype=np.float32))
    arrays = {
        "token_id": ("int32", [1, 2, 3, 4, 2, 5]),
        "time_min": ("float32", [0, 5, 10, 10, 15, 20]),
        "value": ("float32", [0] * 6),
        "has_value": ("uint8", [0] * 6),
        "token_kind": ("uint8", [TOKEN_KINDS["boundary"], TOKEN_KINDS["medication_context"],
                                    TOKEN_KINDS["clinical_outcome"], TOKEN_KINDS["clinical_outcome"],
                                    TOKEN_KINDS["medication_context"], TOKEN_KINDS["clinical_outcome"]]),
        "outcome_class": ("int16", [-1, -1, 1, 2, -1, 0]),
    }
    for name, (dtype, values) in arrays.items():
        np.asarray(values, dtype=dtype).tofile(root / f"{name}.bin")


class EventSequenceCorrectness(unittest.TestCase):
    def test_validation_plateau_scheduler_reaches_late_finetuning_lr(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        scheduler = WarmupPlateauScheduler(
            optimizer, warmup_steps=2, min_lr=1e-6, patience=2, factor=0.1)
        scheduler.step(); scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-3)
        scheduler.step_validation(1.0)
        scheduler.step_validation(1.1)
        self.assertTrue(scheduler.step_validation(1.2))
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)

    def test_sparse_clock_tokens_do_not_restore_five_minute_grid(self):
        records = [
            Record(0.0, "<BOS>", TOKEN_KINDS["boundary"]),
            Record(1440.0, "<EPISODE_END>", TOKEN_KINDS["boundary"]),
        ]
        result = _insert_clock_tokens(records, interval_min=360, max_per_gap=16)
        self.assertEqual([item.time_min for item in result], [0, 360, 720, 1080, 1440])
        self.assertTrue(all(item.kind == TOKEN_KINDS["clock"] for item in result[1:-1]))

    def test_phase_summary_tokens_are_explicit_context_not_targets(self):
        records = [
            Record(0.0, "<BOS>", TOKEN_KINDS["boundary"]),
            Record(10.0, "event:surgery_start", TOKEN_KINDS["clinical_outcome"], 1),
            Record(20.0, "<EPISODE_END>", TOKEN_KINDS["boundary"]),
        ]
        result = _insert_phase_summary_tokens(records)
        tokens = [item.token for item in result]
        self.assertIn("phase_summary:preop", tokens)
        self.assertIn("phase_summary:surgery", tokens)
        self.assertTrue(all(item.outcome < 0 for item in result if item.token.startswith("phase_summary:")))

    def test_expanded_event_metrics_are_finite(self):
        logits = torch.tensor([[4.0, 1.0, 0.0], [0.0, 2.0, 3.0], [1.0, 3.0, 0.0]])
        targets = torch.tensor([[1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=torch.bool)
        report = event_metric_report(logits, targets, ["a", "b", "c"])
        for key in ("micro_auprc", "macro_auprc", "mrr", "brier", "ece"):
            self.assertTrue(np.isfinite(report[key]))
        self.assertEqual(report["event_targets"], 3)

    def test_semantic_family_hits_distinguish_exact_event(self):
        names = ["map_hypotension", "severe_map_hypotension", "tachycardia"]
        ids = family_ids(names)
        targets = torch.tensor([[True, False, False], [False, False, True]])
        predictions = torch.tensor([[1], [0]])
        any_hit, all_hit = family_hit_counts(targets, predictions, ids)
        self.assertEqual(any_hit, 1)
        self.assertEqual(all_hit, 1)

    def test_external_medication_mapping_is_conservative_and_generic(self):
        resolver = MedicationResolver({
            "med:fentanyl": 1,
            "med:insulin": 2,
            "med:insulin lispro": 3,
            "med:salbutamol": 4,
            "med:epinephrine": 5,
            "med:norepinephrine": 6,
        })
        self.assertEqual(
            resolver.resolve("FENTANYL CITRATE (PF) 100 MCG/2ML IJ SOLN"),
            ("med:fentanyl", "generic_prefix"))
        self.assertEqual(
            resolver.resolve("INSULIN LISPRO (HUMAN) 100 UNIT/ML"),
            ("med:insulin lispro", "generic_prefix"))
        self.assertEqual(
            resolver.resolve("ALBUTEROL SULFATE 2.5 MG/3ML"),
            ("med:salbutamol", "explicit_alias"))
        self.assertEqual(
            resolver.resolve("Norepinephrine"),
            ("med:norepinephrine", "exact"))
        self.assertEqual(resolver.resolve("MODEL IMS TEMPLATE"), (None, "unmatched"))

    def test_bucket_sampler_exact_batch_resume(self):
        class VariableLength:
            def __len__(self):
                return 101

            @staticmethod
            def sequence_length(index):
                return (index * 17) % 31 + 1

        dataset = VariableLength()
        full_sampler = LengthSortedBatchSampler(
            dataset, batch_size=8, seed=7, num_buckets=4)
        full_sampler.set_epoch_batch(3, 0)
        full = list(full_sampler)
        resumed_sampler = LengthSortedBatchSampler(
            dataset, batch_size=8, seed=7, num_buckets=4)
        resumed_sampler.set_epoch_batch(3, 5)
        self.assertEqual(list(resumed_sampler), full[5:])
        self.assertEqual(len(resumed_sampler), len(full) - 5)

    def test_context_is_input_only_and_tied_outcomes_form_a_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_fixture(Path(tmp))
            dataset = EventSequenceDataset(tmp, block_size=16)
            self.assertEqual(dataset.sequence_length(0), 6)
            sample = dataset[0]
            # BOS at t=0 and medication at t=5 both target the next strictly
            # later outcome group {a,b}; the medication is never a target.
            self.assertEqual(sample["target_set"][0].tolist(), [0, 1, 1])
            self.assertEqual(sample["target_set"][1].tolist(), [0, 1, 1])
            self.assertFalse(bool(sample["loss_mask"][2]))
            self.assertTrue(bool(sample["loss_mask"][3]))
            self.assertEqual(sample["target_set"][3].tolist(), [1, 0, 0])
            self.assertAlmostEqual(float(sample["target_dt_hours"][1]), 5 / 60)
            # Six-minute trajectory from t=5 contains the tied t=10 outcomes;
            # no current/same-time token is allowed to label itself.
            self.assertEqual(sample["trajectory_target"][1, 0].tolist(), [0, 1, 1])
            self.assertEqual(sample["trajectory_target"][3, 0].tolist(), [0, 0, 0])

    def test_runtime_vocabulary_ablation_and_dynamic_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_fixture(Path(tmp))
            dataset = EventSequenceDataset(
                tmp, block_size=8, dynamic_windows=True, seed=3, outcome_limit=2)
            self.assertEqual(dataset.num_outcomes, 2)
            self.assertEqual(dataset.meta["outcome_vocabulary"], ["a", "b"])
            before = dataset.window_start.copy()
            dataset.set_epoch(2)
            self.assertEqual(len(before), len(dataset.window_start))
            self.assertGreaterEqual(dataset.sequence_length(0), 6)

    def test_measurement_confirmation_and_observed_response(self):
        transitions = state_transitions(
            np.array([0, 5, 10, 15], dtype=np.float32),
            np.array([70, 60, 58, 70], dtype=np.float32),
            lambda value: value < 65, lambda value: value < 55,
            lambda value: value >= 65, "low", "recovered", lambda value: (65 - value) / 10,
        )
        self.assertEqual([item.name for item in transitions], ["low"])
        # A single normal value is intentionally insufficient for recovery.
        status, when = observed_vasopressor_response(
            5, np.array([5, 10, 15]), np.array([1, 0, 0]))
        self.assertEqual(status, "recovered")
        self.assertEqual(when, 15)

    def test_model_loss_is_finite_and_static_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_fixture(Path(tmp))
            dataset = EventSequenceDataset(tmp, block_size=16)
            batch = collate_event_sequences([dataset[0]])
            model = EventMAOMAO(dataset.num_tokens, dataset.num_outcomes, dataset.num_static,
                              hidden_dim=32, num_layers=1, num_heads=4, ffn_dim=64,
                              dropout=0).eval()
            logits = model(batch).logits
            loss = event_time_loss(logits, batch["target_set"],
                                   batch["target_dt_hours"], batch["loss_mask"])
            self.assertTrue(torch.isfinite(loss["loss"]))
            loss["loss"].backward()
            self.assertIsNotNone(model.static_projection[0].weight.grad)
            changed = {key: value.clone() for key, value in batch.items()}
            changed["static"][:, 0] += 1
            with torch.no_grad():
                self.assertFalse(torch.allclose(logits, model(changed).logits))

    def test_decoupled_rate_head_and_enhanced_time_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_fixture(Path(tmp))
            dataset = EventSequenceDataset(tmp, block_size=16)
            batch = collate_event_sequences([dataset[0]])
            model = EventMAOMAO(
                dataset.num_tokens, dataset.num_outcomes, dataset.num_static,
                hidden_dim=32, num_layers=1, num_heads=4, ffn_dim=64, dropout=0,
                decoupled_time_head=True, enhanced_time_encoding=True,
            )
            output = model(batch)
            self.assertEqual(output.log_total_rate.shape, output.logits.shape[:2])
            result = event_time_loss(
                output.logits, batch["target_set"], batch["target_dt_hours"],
                batch["loss_mask"], batch["time_mask"],
                log_total_rate=output.log_total_rate,
            )
            self.assertTrue(torch.isfinite(result["loss"]))
            result["loss"].backward()
            self.assertIsNotNone(model.time_head.weight.grad)

    def test_v4_ontology_and_multitask_heads(self):
        self.assertGreaterEqual(len(OUTCOMES), 100)
        events = extract_measurement_events({
            "labs:creatinine": (
                np.array([0, 60, 120], dtype=np.float32),
                np.array([0.8, 1.3, 2.5], dtype=np.float32),
            ),
            "labs:potassium": (
                np.array([0, 60], dtype=np.float32),
                np.array([4.0, 5.3], dtype=np.float32),
            ),
            "labs:lactate": (
                np.array([0, 60], dtype=np.float32),
                np.array([1.2, 4.2], dtype=np.float32),
            ),
        })
        names = {event.name for event in events}
        self.assertIn("aki_stage_1_signal", names)
        self.assertIn("aki_stage_3_signal", names)
        self.assertIn("severe_hyperkalemia", names)
        self.assertIn("severe_hyperlactatemia", names)

        with tempfile.TemporaryDirectory() as tmp:
            event_fixture(Path(tmp))
            dataset = EventSequenceDataset(tmp, block_size=16)
            batch = collate_event_sequences([dataset[0]])
            model = EventMAOMAO(
                dataset.num_tokens, dataset.num_outcomes, dataset.num_static,
                hidden_dim=32, num_layers=1, num_heads=4, ffn_dim=64,
                num_trajectory_horizons=1, lognormal_time_head=True,
            )
            output = model(batch)
            self.assertEqual(output.trajectory_logits.shape[-2:], (1, dataset.num_outcomes))
            result = event_time_loss(
                output.logits, batch["target_set"], batch["target_dt_hours"],
                batch["loss_mask"], batch["time_mask"],
                time_mu=output.time_mu, time_log_sigma=output.time_log_sigma,
                trajectory_logits=output.trajectory_logits,
                trajectory_target=batch["trajectory_target"],
                trajectory_mask=batch["trajectory_mask"], trajectory_weight=0.5,
            )
            self.assertTrue(torch.isfinite(result["loss"]))
            masked = model(batch, causal=False, return_token_logits=True)
            self.assertEqual(masked.token_logits.shape[-1], dataset.num_tokens)

    def test_block_causal_relative_bias_and_hierarchical_time_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_fixture(Path(tmp))
            dataset = EventSequenceDataset(tmp, block_size=16)
            batch = collate_event_sequences([dataset[0]])
            outcome_family_ids = torch.tensor([0, 1, 1])
            model = EventMAOMAO(
                dataset.num_tokens, dataset.num_outcomes, dataset.num_static,
                hidden_dim=32, num_layers=1, num_heads=4, ffn_dim=64, dropout=0,
                lognormal_time_head=True,
                outcome_family_ids=outcome_family_ids,
                same_time_block_causal=True,
                relative_time_attention=True,
                event_conditioned_time_head=True,
            )
            mask = model._attention_mask(batch["time_min"], causal=True).reshape(
                1, 4, 6, 6)
            # Positions 2 and 3 occur together at minute 10 and are mutually hidden.
            self.assertTrue(torch.isneginf(mask[0, :, 2, 3]).all())
            self.assertTrue(torch.isneginf(mask[0, :, 3, 2]).all())
            self.assertTrue(torch.isfinite(mask[0, :, 3, 3]).all())
            output = model(batch)
            self.assertEqual(output.family_logits.shape[-1], 2)
            self.assertEqual(output.family_time_mu.shape[-1], 2)
            result = event_time_loss(
                output.logits, batch["target_set"], batch["target_dt_hours"],
                batch["loss_mask"], batch["time_mask"],
                time_mu=output.time_mu, time_log_sigma=output.time_log_sigma,
                family_logits=output.family_logits,
                outcome_family_ids=outcome_family_ids, family_weight=0.5,
                family_time_mu=output.family_time_mu,
                family_time_log_sigma=output.family_time_log_sigma,
            )
            self.assertTrue(torch.isfinite(result["loss"]))
            self.assertTrue(torch.isfinite(result["family_loss"]))

    def test_dual_timescale_event_specific_hazard_head_has_gradients(self):
        model = EventMAOMAO(
            num_tokens=8, num_outcomes=3, num_static=2, hidden_dim=32,
            num_layers=1, num_heads=4, ffn_dim=64, lognormal_time_head=False,
            outcome_family_ids=torch.tensor([0, 0, 1]),
            dual_timescale_time_head=True, fine_time_bins=24, long_time_bins=8)
        batch = {
            "token_id": torch.tensor([[1, 2, 3, 4]]),
            "token_kind": torch.ones(1, 4, dtype=torch.long),
            "value": torch.zeros(1, 4), "has_value": torch.zeros(1, 4),
            "time_min": torch.tensor([[0., 5., 10., 15.]]),
            "gap_min": torch.tensor([[0., 5., 5., 5.]]),
            "static": torch.zeros(1, 2),
            "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        }
        output = model(batch)
        target = torch.zeros(1, 4, 3); target[0, 2, 1] = 1
        dt = torch.ones(1, 4); loss_mask = torch.zeros(1, 4, dtype=torch.bool)
        time_mask = torch.zeros(1, 4, dtype=torch.bool)
        loss_mask[0, 2] = True; time_mask[0, 2] = True
        result = event_time_loss(
            output.logits, target, dt, loss_mask, time_mask,
            fine_hazard_logits=output.fine_hazard_logits,
            long_hazard_logits=output.long_hazard_logits,
            tail_mu=output.tail_mu, tail_log_sigma=output.tail_log_sigma)
        self.assertEqual(output.fine_hazard_logits.shape, (1, 4, 3, 24))
        self.assertEqual(output.long_hazard_logits.shape, (1, 4, 3, 8))
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertIsNotNone(model.dual_hazard_head.weight.grad)

    def test_family_head_and_clock_phase_context_can_be_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_fixture(Path(tmp))
            dataset = EventSequenceDataset(tmp, block_size=16)
            batch = collate_event_sequences([dataset[0]])
            model = EventMAOMAO(
                dataset.num_tokens, dataset.num_outcomes, dataset.num_static,
                hidden_dim=32, num_layers=1, num_heads=4, ffn_dim=64,
                lognormal_time_head=True,
                outcome_family_ids=torch.tensor([0, 1, 1]),
                use_family_head=False,
                clock_phase_context=False,
                phase_memory=False,
            )
            model.clock_phase_token_ids = torch.tensor([int(batch["token_id"][0, 0])])
            output = model(batch)
            self.assertIsNone(output.family_logits)
            self.assertTrue(torch.isfinite(output.logits).all())


if __name__ == "__main__":
    unittest.main()
