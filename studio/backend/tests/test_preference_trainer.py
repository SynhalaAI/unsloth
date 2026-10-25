# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for Studio's DPO/ORPO/CPO trainer wiring.

TRL is faked through sys.modules so these run without torch: the point is the contract with
TRL 0.23.1 (config owns beta/max_length/max_prompt_length, the tokenizer is
`processing_class=`, PatchFastRL runs before the class import) and the dataset conversion, not
the training math.
"""

import sys
import types
from dataclasses import dataclass

import pytest

from core.training import preference_trainer as pt


class FakeDataset:
    """Map-style dataset: __getitem__ for the probe, map() for the chat-template pass."""

    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.column_names = list(self.rows[0])
        self.map_calls = 0

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)

    def add_column(self, name, column):
        for row, value in zip(self.rows, column):
            row[name] = value
        self.column_names = list(self.rows[0])
        return self

    def map(self, function, **kwargs):
        self.map_calls += 1
        self.rows = [function(row) for row in self.rows]
        self.column_names = list(self.rows[0])
        return self


class FakeStreamingDataset:
    """Iterable dataset: no add_column, so reference-logprob precomputation cannot work."""

    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.column_names = list(self.rows[0])

    def __getitem__(self, index):
        return self.rows[index]

    def map(self, function, **kwargs):
        return self


class FakeTokenizer:
    def __init__(self, chat_template = None, pad_token = "<pad>", eos_token = "<eos>"):
        self.chat_template = chat_template
        self.pad_token = pad_token
        self.eos_token = eos_token

    @property
    def pad_token_id(self):
        return None if self.pad_token is None else 0

    def apply_chat_template(
        self, messages, tokenize = False, add_generation_prompt = False, **kwargs
    ):
        text = "".join(f"<{message['role']}>{message['content']}</>" for message in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


class FakeModel:
    def __init__(self, max_seq_length = None):
        if max_seq_length is not None:
            self.max_seq_length = max_seq_length


PREFERENCE_ROWS = [{"prompt": "2+2?", "chosen": "4", "rejected": "5"}]


def _install_fake_unsloth(monkeypatch, events):
    module = types.ModuleType("unsloth")

    def PatchFastRL(method):
        events.append(("patch", method))

    module.PatchFastRL = PatchFastRL
    monkeypatch.setitem(sys.modules, "unsloth", module)


def _make_trainer(name, *, tokenizer_kwarg = False, ref_model_kwarg = False):
    """Mirror TRL's signatures: only DPOTrainer takes ref_model, only TRL < 0.19 takes tokenizer."""

    if tokenizer_kwarg:

        class _OldTrainer:
            def __init__(
                self,
                model = None,
                args = None,
                train_dataset = None,
                eval_dataset = None,
                tokenizer = None,
            ):
                self.model = model
                self.args = args
                self.train_dataset = train_dataset
                self.eval_dataset = eval_dataset
                self.tokenizer = tokenizer
                self.captured = {
                    "model": model,
                    "args": args,
                    "train_dataset": train_dataset,
                    "eval_dataset": eval_dataset,
                    "tokenizer": tokenizer,
                }

        return _OldTrainer

    if ref_model_kwarg:

        class _DPOTrainer:
            def __init__(
                self,
                model = None,
                args = None,
                train_dataset = None,
                eval_dataset = None,
                processing_class = None,
                ref_model = None,
            ):
                self.model = model
                self.args = args
                self.train_dataset = train_dataset
                self.eval_dataset = eval_dataset
                self.processing_class = processing_class
                self.captured = {
                    "model": model,
                    "args": args,
                    "train_dataset": train_dataset,
                    "eval_dataset": eval_dataset,
                    "processing_class": processing_class,
                    "ref_model": ref_model,
                }

        return _DPOTrainer

    class _PreferenceTrainer:
        def __init__(
            self,
            model = None,
            args = None,
            train_dataset = None,
            eval_dataset = None,
            processing_class = None,
        ):
            self.model = model
            self.args = args
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.processing_class = processing_class
            self.captured = {
                "model": model,
                "args": args,
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset,
                "processing_class": processing_class,
            }

    return _PreferenceTrainer


def _install_fake_trl(monkeypatch, events, *, tokenizer_kwarg = False):
    @dataclass
    class DPOConfig:
        output_dir: str = "."
        per_device_train_batch_size: int = 1
        learning_rate: float = 5e-6
        beta: float = 0.1
        max_length: int = 1024
        max_prompt_length: int = 512
        precompute_ref_log_probs: bool = False
        remove_unused_columns: bool = True
        include_num_input_tokens_seen: bool = True

    @dataclass
    class ORPOConfig:
        output_dir: str = "."
        per_device_train_batch_size: int = 1
        learning_rate: float = 5e-6
        beta: float = 0.1
        max_length: int = 1024
        max_prompt_length: int = 512
        remove_unused_columns: bool = True
        include_num_input_tokens_seen: bool = True

    @dataclass
    class CPOConfig:
        output_dir: str = "."
        per_device_train_batch_size: int = 1
        learning_rate: float = 5e-6
        beta: float = 0.1
        max_length: int = 1024
        max_prompt_length: int = 512
        cpo_alpha: float = 1.0
        remove_unused_columns: bool = True
        include_num_input_tokens_seen: bool = True

    classes = {
        "DPOConfig": DPOConfig,
        "DPOTrainer": _make_trainer(
            "DPO", tokenizer_kwarg = tokenizer_kwarg, ref_model_kwarg = True
        ),
        "ORPOConfig": ORPOConfig,
        "ORPOTrainer": _make_trainer("ORPO", tokenizer_kwarg = tokenizer_kwarg),
        "CPOConfig": CPOConfig,
        "CPOTrainer": _make_trainer("CPO", tokenizer_kwarg = tokenizer_kwarg),
    }

    module = types.ModuleType("trl")

    def _resolve(name):
        if name in classes:
            events.append(("import", name))
            return classes[name]
        raise AttributeError(name)

    module.__dict__["__getattr__"] = _resolve
    monkeypatch.setitem(sys.modules, "trl", module)
    return classes


def _setup(monkeypatch, method, **overrides):
    events = []
    _install_fake_unsloth(monkeypatch, events)
    classes = _install_fake_trl(monkeypatch, events, tokenizer_kwarg = overrides.pop("old_trl", False))
    kwargs = {
        "model": FakeModel(max_seq_length = 4096),
        "processing_class": FakeTokenizer(),
        "train_dataset": FakeDataset(PREFERENCE_ROWS),
        "config_args": {"output_dir": "out", "per_device_train_batch_size": 2},
        "beta": 0.2,
        "max_prompt_length": 512,
        "max_length": 2048,
    }
    kwargs.update(overrides)
    trainer = pt.setup_preference_trainer(method, **kwargs)
    return trainer, events, classes


def test_dpo_puts_beta_and_lengths_on_the_config(monkeypatch):
    trainer, _, _ = _setup(
        monkeypatch,
        "dpo",
        config_args = {
            "output_dir": "out",
            "per_device_train_batch_size": 2,
            # SFT-only keys: SFTConfig takes them, DPOConfig does not.
            "packing": True,
            "max_seq_length": 2048,
            "dataset_text_field": "text",
        },
    )

    config = trainer.args
    assert config.beta == 0.2
    assert config.max_length == 2048
    assert config.max_prompt_length == 512
    assert config.per_device_train_batch_size == 2
    assert config.output_dir == "out"
    assert not hasattr(config, "packing")
    assert not hasattr(config, "dataset_text_field")


def test_dpo_uses_processing_class_and_a_null_reference(monkeypatch):
    tokenizer = FakeTokenizer()
    trainer, _, _ = _setup(
        monkeypatch,
        "dpo",
        processing_class = tokenizer,
        old_trl = False,
    )

    assert trainer.captured["processing_class"] is tokenizer
    assert "tokenizer" not in trainer.captured
    assert trainer.captured["ref_model"] is None


def test_unsloth_patch_runs_before_the_trainer_import(monkeypatch):
    _, events, _ = _setup(monkeypatch, "dpo")

    assert events[0] == ("patch", "DPO")
    trainer_imports = [index for index, event in enumerate(events) if event[0] == "import"]
    assert trainer_imports and min(trainer_imports) > 0


def test_unknown_method_is_refused(monkeypatch):
    events = []
    _install_fake_unsloth(monkeypatch, events)
    _install_fake_trl(monkeypatch, events)

    with pytest.raises(ValueError, match = "Unknown preference training method"):
        pt.setup_preference_trainer(
            "KTO",
            model = FakeModel(),
            processing_class = FakeTokenizer(),
            train_dataset = FakeDataset(PREFERENCE_ROWS),
        )


def test_preference_method_on_an_sft_dataset_is_refused(monkeypatch):
    events = []
    _install_fake_unsloth(monkeypatch, events)
    _install_fake_trl(monkeypatch, events)
    sft_dataset = FakeDataset([{"instruction": "2+2?", "output": "4"}])

    with pytest.raises(ValueError, match = "Dataset mismatch"):
        pt.setup_preference_trainer(
            "DPO",
            model = FakeModel(),
            processing_class = FakeTokenizer(),
            train_dataset = sft_dataset,
        )


def test_mismatched_eval_dataset_is_refused(monkeypatch):
    events = []
    _install_fake_unsloth(monkeypatch, events)
    _install_fake_trl(monkeypatch, events)

    with pytest.raises(ValueError, match = "Eval dataset mismatch"):
        pt.setup_preference_trainer(
            "DPO",
            model = FakeModel(),
            processing_class = FakeTokenizer(),
            train_dataset = FakeDataset(PREFERENCE_ROWS),
            eval_dataset = FakeDataset([{"instruction": "2+2?", "output": "4"}]),
        )


def test_max_length_is_clamped_to_the_model_context(monkeypatch):
    trainer, _, _ = _setup(
        monkeypatch,
        "dpo",
        model = FakeModel(max_seq_length = 512),
        max_length = 4096,
        max_prompt_length = 1024,
    )

    assert trainer.args.max_length == 512
    assert trainer.args.max_prompt_length == 256


def test_full_finetune_dpo_precomputes_the_reference_logprobs(monkeypatch):
    monkeypatch.setitem(sys.modules, "peft", types.ModuleType("peft"))

    trainer, _, _ = _setup(monkeypatch, "dpo", model = FakeModel(max_seq_length = 2048))

    assert trainer.args.precompute_ref_log_probs is True
    assert trainer.captured["ref_model"] is None


def test_peft_dpo_keeps_the_adapter_disabled_reference(monkeypatch):
    peft_module = types.ModuleType("peft")

    class PeftModel:
        pass

    peft_module.PeftModel = PeftModel
    monkeypatch.setitem(sys.modules, "peft", peft_module)

    trainer, _, _ = _setup(monkeypatch, "dpo", model = PeftModel())

    assert trainer.args.precompute_ref_log_probs is False
    assert trainer.captured["ref_model"] is None


def test_full_finetune_dpo_refuses_a_streaming_dataset(monkeypatch):
    monkeypatch.setitem(sys.modules, "peft", types.ModuleType("peft"))

    with pytest.raises(ValueError, match = "map-style dataset"):
        _setup(
            monkeypatch,
            "dpo",
            train_dataset = FakeStreamingDataset(PREFERENCE_ROWS),
            model = FakeModel(max_seq_length = 2048),
        )


@pytest.mark.parametrize("method", ["ORPO", "CPO"])
def test_orpo_and_cpo_disable_remove_unused_columns(monkeypatch, method):
    trainer, _, _ = _setup(monkeypatch, method)

    # Their DPODataCollatorWithPadding needs the prompt/chosen/rejected columns. TRL flips this
    # itself but warns on every run, so Studio sets it explicitly.
    assert trainer.args.remove_unused_columns is False


@pytest.mark.parametrize("method", ["DPO", "ORPO", "CPO"])
def test_token_tracking_is_off_for_preference_methods(monkeypatch, method):
    trainer, _, _ = _setup(
        monkeypatch,
        method,
        config_args = {
            "output_dir": "out",
            # The shared SFT config turns this on. A preference batch has no `input_ids`, so HF
            # would warn on every step and leave the counter at 0.
            "include_num_input_tokens_seen": True,
        },
    )

    assert trainer.args.include_num_input_tokens_seen is False


def test_dpo_leaves_remove_unused_columns_to_trl(monkeypatch):
    trainer, _, _ = _setup(monkeypatch, "dpo")

    # DPOTrainer overrides the signature columns instead, so it must keep the Trainer default.
    assert trainer.args.remove_unused_columns is True


@pytest.mark.parametrize("method", ["ORPO", "CPO"])
def test_orpo_and_cpo_take_no_reference_model(monkeypatch, method):
    trainer, _, _ = _setup(monkeypatch, method, cpo_alpha = 0.7)

    assert "ref_model" not in trainer.captured
    if method == "CPO":
        assert trainer.args.cpo_alpha == 0.7
    else:
        assert not hasattr(trainer.args, "cpo_alpha")


def test_preference_trainer_takes_a_tokenizer_on_old_trl(monkeypatch):
    tokenizer = FakeTokenizer()
    trainer, _, _ = _setup(monkeypatch, "cpo", processing_class = tokenizer, old_trl = True)

    assert trainer.captured["tokenizer"] is tokenizer
    assert "processing_class" not in trainer.captured


def test_pad_token_falls_back_to_eos(monkeypatch):
    tokenizer = FakeTokenizer(pad_token = None)

    trainer, _, _ = _setup(monkeypatch, "dpo", processing_class = tokenizer)

    assert tokenizer.pad_token == tokenizer.eos_token
    assert trainer.captured["processing_class"] is tokenizer


def test_string_rows_become_chat_messages():
    dataset = FakeDataset(
        [
            {
                "system": "Be brief.",
                "prompt": "Be brief.\n\n2+2?",
                "chosen": "4",
                "rejected": "5",
            }
        ]
    )

    converted = pt.prepare_preference_dataset(dataset, FakeTokenizer(chat_template = "template"))

    assert converted.map_calls == 1
    assert converted[0]["prompt"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "2+2?"},
    ]
    assert converted[0]["chosen"] == [{"role": "assistant", "content": "4"}]
    assert converted[0]["rejected"] == [{"role": "assistant", "content": "5"}]


def test_already_templated_prompts_are_left_alone():
    dataset = FakeDataset(
        [{"prompt": "<user>2+2?</>", "chosen": "4", "rejected": "5"}]
    )

    converted = pt.prepare_preference_dataset(dataset, FakeTokenizer(chat_template = "template"))

    assert converted.map_calls == 0
    assert converted[0]["prompt"] == "<user>2+2?</>"


def test_base_models_without_a_chat_template_are_left_alone():
    dataset = FakeDataset(PREFERENCE_ROWS)

    converted = pt.prepare_preference_dataset(dataset, FakeTokenizer(chat_template = None))

    assert converted.map_calls == 0
    assert converted[0]["prompt"] == "2+2?"


def test_conversational_datasets_are_left_alone():
    dataset = FakeDataset(
        [
            {
                "prompt": [{"role": "user", "content": "2+2?"}],
                "chosen": [{"role": "assistant", "content": "4"}],
                "rejected": [{"role": "assistant", "content": "5"}],
            }
        ]
    )

    converted = pt.prepare_preference_dataset(dataset, FakeTokenizer(chat_template = "template"))

    assert converted.map_calls == 0


def test_preference_dataset_mismatch_requires_chosen_and_rejected():
    class _Columns:
        def __init__(self, columns):
            self.column_names = columns

    sft_rows = _Columns(["instruction", "output"])
    preference_rows = _Columns(["prompt", "chosen", "rejected"])

    for method in ("DPO", "ORPO", "CPO", "orpo"):
        message = pt.preference_dataset_mismatch(method, sft_rows)
        assert message is not None
        assert message.startswith("Dataset mismatch:")
        assert pt.preference_dataset_mismatch(method, preference_rows) is None

    # SFT (and an unset method) never consults the dataset shape.
    assert pt.preference_dataset_mismatch("SFT", sft_rows) is None
    assert pt.preference_dataset_mismatch(None, sft_rows) is None

    # The trainer hands over the {"dataset": ...} wrapper, so it must be unwrapped.
    assert pt.preference_dataset_mismatch("ORPO", {"dataset": preference_rows}) is None
    assert pt.preference_dataset_mismatch("ORPO", {"dataset": sft_rows}) is not None


def test_orpo_falls_back_to_trl_experimental(monkeypatch):
    events = []
    _install_fake_unsloth(monkeypatch, events)

    @dataclass
    class ORPOConfig:
        output_dir: str = "."
        beta: float = 0.1
        max_length: int = 1024
        max_prompt_length: int = 512
        remove_unused_columns: bool = True
        include_num_input_tokens_seen: bool = True

    class ORPOTrainer:
        def __init__(self, model = None, args = None, train_dataset = None, processing_class = None):
            self.args = args
            self.captured = {"processing_class": processing_class}

    # TRL releases that moved ORPO out of the top level only expose trl.experimental.orpo.
    trl = types.ModuleType("trl")
    trl.__path__ = []
    experimental = types.ModuleType("trl.experimental")
    experimental.__path__ = []
    orpo = types.ModuleType("trl.experimental.orpo")
    orpo.ORPOTrainer = ORPOTrainer
    orpo.ORPOConfig = ORPOConfig
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "trl.experimental", experimental)
    monkeypatch.setitem(sys.modules, "trl.experimental.orpo", orpo)

    tokenizer = FakeTokenizer()
    trainer = pt.setup_preference_trainer(
        "ORPO",
        model = FakeModel(),
        processing_class = tokenizer,
        train_dataset = FakeDataset(PREFERENCE_ROWS),
    )

    assert isinstance(trainer, ORPOTrainer)
    assert trainer.captured["processing_class"] is tokenizer
    assert events[0] == ("patch", "ORPO")


def test_worker_config_forwards_the_preference_settings():
    """The route's whitelist, the worker config builder and the worker call must all keep the
    four fields, or the trainer silently runs SFT on chosen/rejected columns."""
    training = pytest.importorskip("core.training.training")

    config = training._build_training_worker_config(
        {
            "model_name": "unsloth/Qwen3-4B",
            "training_method": "ORPO",
            "dpo_beta": 0.3,
            "cpo_alpha": 0.9,
            "max_prompt_length": 128,
        }
    )

    assert config["training_method"] == "ORPO"
    assert config["dpo_beta"] == 0.3
    assert config["cpo_alpha"] == 0.9
    assert config["max_prompt_length"] == 128
    assert config["format_type"] == ""
