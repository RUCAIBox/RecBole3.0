from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from recbole3.dataset import ITEM_ID, LABEL, SEEN_ITEM_IDS, USER_ID
from recbole3.evaluation import EvalConfig
from recbole3.model.rqvae.trainer import RQVAETrainer
from recbole3.model.minionerec.config import MiniOneRecConfig
from recbole3.model.minionerec.data import (
    MINIONEREC_SEQREC_INSTRUCTION,
    MiniOneRecSIDCodec,
    MiniOneRecSFTDataset,
    build_minionerec_rl_datasets,
    build_item_alignment_examples,
    build_sequence_sft_examples,
    load_minionerec_sid_codec,
)
from recbole3.model.minionerec.logits import (
    MiniOneRecConstrainedLogitsProcessor,
    build_minionerec_prefix_allowed_tokens,
)
from recbole3.model.minionerec.pipeline import MiniOneRecPipeline
from recbole3.model.minionerec.rewards import (
    build_minionerec_reward_functions,
    normalize_minionerec_ranking_text,
    normalize_minionerec_rule_text,
)
from recbole3.model.minionerec.trainer import (
    MiniOneRecGenerationCollator,
    MiniOneRecGenerationRetrievalModel,
    MiniOneRecTrainer,
    _select_generated_item_ids,
)
from recbole3.run import run_experiment
from tests.test_helpers import StubDataset, StubDatasetConfig, ensure_stub_tables


class FakeTokenizer:
    bos_token_id = 101
    eos_token_id = 102
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.calls.append(str(text))
        return [self.bos_token_id, *(ord(char) % 97 + 3 for char in text), self.eos_token_id]

    def __call__(self, text: str):
        class Tokenized:
            def __init__(self, input_ids: list[int]) -> None:
                self.input_ids = input_ids

        return Tokenized(self.encode(text))


def _sid_file(tmp_path, num_items: int = 8) -> str:
    path = tmp_path / "item.index.json"
    payload = {str(item_id): [f"<1_{item_id}>", f"<2_{item_id}>", f"<3_{item_id}>"] for item_id in range(num_items)}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _prepared_stub_dataset():
    return StubDataset(StubDatasetConfig()).prepare(eval_config=EvalConfig(protocol="full"))


def test_minionerec_sid_codec_concatenates_original_tokens(tmp_path) -> None:
    codec = MiniOneRecSIDCodec.from_file(_sid_file(tmp_path), num_items=8)

    assert codec.item_sid(3) == "<1_3><2_3><3_3>"
    assert codec.decode_sid("<1_3><2_3><3_3>") == 3
    assert "<2_7>" in codec.all_tokens


def test_minionerec_sid_codec_remaps_raw_source_item_ids(tmp_path) -> None:
    prepared = _prepared_stub_dataset()
    prepared._item_id_map = {100 + item_id: item_id for item_id in range(prepared.get_num_items())}
    path = tmp_path / "raw.index.json"
    payload = {str(100 + item_id): [f"<a_{item_id}>", f"<b_{item_id}>"] for item_id in range(prepared.get_num_items())}
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = MiniOneRecConfig(sid_file=str(path), sid_file_item_id_space="raw")

    codec = load_minionerec_sid_codec(config, prepared)

    assert codec.item_sid(3) == "<a_3><b_3>"
    assert codec.decode_sid("<a_3><b_3>") == 3


def test_minionerec_sid_codec_rejects_duplicate_sids_by_default(tmp_path) -> None:
    path = tmp_path / "duplicate.index.json"
    path.write_text(
        json.dumps({"0": ["<a_0>", "<b_0>"], "1": ["<a_0>", "<b_0>"], "2": ["<a_2>", "<b_2>"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate SID strings"):
        MiniOneRecSIDCodec.from_file(str(path), num_items=3)


def test_minionerec_sid_codec_can_keep_duplicate_sid_aliases_when_explicit(tmp_path) -> None:
    path = tmp_path / "duplicate.index.json"
    path.write_text(
        json.dumps({"0": ["<a_0>", "<b_0>"], "1": ["<a_0>", "<b_0>"], "2": ["<a_2>", "<b_2>"]}),
        encoding="utf-8",
    )
    codec = MiniOneRecSIDCodec.from_file(str(path), num_items=3, allow_duplicate_sid_aliases=True)

    assert codec.decode_sid("<a_0><b_0>") == 0
    assert codec.decode_sid_candidates("<a_0><b_0>") == (0, 1)

    predictions, stats = _select_generated_item_ids(["<a_0><b_0>"], codec, excluded={0}, k=2)

    assert predictions == [1, -1]
    assert stats["valid_generations"] == 1
    assert stats["selected_generations"] == 1


def test_rqvae_minionerec_index_output_loads_as_minionerec_sid_codec(tmp_path) -> None:
    prepared = _prepared_stub_dataset()
    payload = RQVAETrainer._to_minionerec_index_json(
        {"0": [1, 2, 3], "1": [1, 2, 3], "2": [4, 5, 6], "3": [7, 8, 9]},
        token_prefixes=("a", "b", "c", "d"),
        token_offset=0,
    )
    path = tmp_path / "item.index.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    codec = load_minionerec_sid_codec(MiniOneRecConfig(sid_file=str(path), require_complete_sid_file=False), prepared)

    assert codec.item_sid(0) == "<a_1><b_2><c_3><d_1>"
    assert codec.item_sid(1) == "<a_1><b_2><c_3><d_2>"
    assert codec.decode_sid("<a_4><b_5><c_6>") == 2


def test_minionerec_prefix_constraint_allows_only_valid_sid_continuations() -> None:
    tokenizer = FakeTokenizer()
    semantic_id = "<1_2><2_2><3_2>"
    prefix_allowed_tokens_fn, prefix_token_count = build_minionerec_prefix_allowed_tokens(
        tokenizer,
        (semantic_id,),
        base_model="toy-model",
        prefix_token_count=4,
    )
    token_ids = tokenizer(f"### Response:\n{semantic_id}\n").input_ids + [tokenizer.eos_token_id]
    vocab_size = max(max(token_ids) + 1, tokenizer.eos_token_id + 1, 128)

    assert prefix_allowed_tokens_fn(0, token_ids[:prefix_token_count]) == [token_ids[prefix_token_count]]
    assert prefix_allowed_tokens_fn(0, [999, 998, 997]) == []

    processor = MiniOneRecConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn,
        num_beams=1,
        prefix_token_count=prefix_token_count,
        eos_token_id=tokenizer.eos_token_id,
    )
    scores = torch.zeros((1, vocab_size), dtype=torch.float32)
    processed = processor(torch.tensor([token_ids[:prefix_token_count]], dtype=torch.long), scores)

    assert torch.isfinite(processed[0]).nonzero(as_tuple=True)[0].tolist() == [token_ids[prefix_token_count]]
    assert processor.stats()["constraint_valid_prefix_checks"] == 1

    invalid_processor = MiniOneRecConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn,
        num_beams=1,
        prefix_token_count=prefix_token_count,
        eos_token_id=tokenizer.eos_token_id,
    )
    with pytest.warns(RuntimeWarning, match="No valid MiniOneRec tokens"):
        invalid_processed = invalid_processor(torch.tensor([[999, 998, 997, 996]], dtype=torch.long), scores)

    assert torch.isfinite(invalid_processed[0]).nonzero(as_tuple=True)[0].tolist() == [tokenizer.eos_token_id]
    assert invalid_processor.stats()["constraint_forced_eos_count"] == 1


def test_minionerec_sequence_examples_use_original_prompt_semantics(tmp_path) -> None:
    prepared = _prepared_stub_dataset()
    config = MiniOneRecConfig(sid_file=_sid_file(tmp_path), history_max_length=2)
    codec = MiniOneRecSIDCodec.from_file(config.sid_file, num_items=prepared.get_num_items())

    train_examples = build_sequence_sft_examples(config, codec, prepared, split="train", eval_prompt=False)
    valid_examples = build_sequence_sft_examples(config, codec, prepared, split="valid", eval_prompt=True)

    assert train_examples[0]["input"] == (
        "The user has interacted with items <1_0><2_0><3_0> in chronological order. "
        "Can you predict the next possible item that the user may expect?"
    )
    assert train_examples[0]["output"] == "<1_1><2_1><3_1>\n"
    assert valid_examples[0]["input"] == (
        "Can you predict the next possible item the user may expect, "
        "given the following chronological interaction history: <1_0><2_0><3_0>, <1_1><2_1><3_1>"
    )
    assert valid_examples[0]["target_item_id"] == 2


def test_minionerec_item_alignment_examples_use_item_titles(tmp_path) -> None:
    prepared = _prepared_stub_dataset()
    config = MiniOneRecConfig(sid_file=_sid_file(tmp_path))
    codec = MiniOneRecSIDCodec.from_file(config.sid_file, num_items=prepared.get_num_items())

    examples = build_item_alignment_examples(config, codec, prepared)

    assert {"input": "Which item has the title: Alpha Quest?", "output": "<1_0><2_0><3_0>\n"} in examples
    assert {"input": 'What is the title of item "<1_0><2_0><3_0>"?', "output": "Alpha Quest\n"} in examples


def test_minionerec_sft_dataset_masks_prompt_tokens() -> None:
    dataset = MiniOneRecSFTDataset(
        [{"input": "history prompt", "output": "<1_0><2_0><3_0>\n"}],
        FakeTokenizer(),
        max_len=512,
    )

    row = dataset[0]

    assert len(row["input_ids"]) == len(row["attention_mask"]) == len(row["labels"])
    assert row["labels"].count(-100) > 0
    assert row["labels"][-1] == FakeTokenizer.eos_token_id


def test_minionerec_rl_dataset_keeps_original_prompt_completion_maps(tmp_path) -> None:
    prepared = _prepared_stub_dataset()
    config = MiniOneRecConfig(
        sid_file=_sid_file(tmp_path),
        history_max_length=2,
        rl_add_item_alignment_tasks=False,
        rl_add_title_sequence_task=False,
    )
    codec = MiniOneRecSIDCodec.from_file(config.sid_file, num_items=prepared.get_num_items())

    rl_data = build_minionerec_rl_datasets(config, codec, prepared)
    first = rl_data.train_dataset[0]

    assert first["prompt"].startswith("### User Input: \nThe user has interacted with items <1_0><2_0><3_0>")
    assert first["completion"] == "<1_1><2_1><3_1>\n"
    assert first["excluded_item_ids"] == (0,)
    assert rl_data.prompt2history[first["prompt"]] == "<1_0><2_0><3_0>"
    assert rl_data.history2target["<1_0><2_0><3_0>"] == "<1_1><2_1><3_1>\n"
    assert rl_data.prompt2excluded_item_ids[first["prompt"]] == (0,)


def test_minionerec_rl_exclude_history_skips_unreachable_repeated_targets(tmp_path) -> None:
    prepared = _prepared_stub_dataset()
    train_frame = prepared.get_train_dataset().frame.reset_index(drop=True).copy()
    repeated_target = pd.DataFrame([{USER_ID: 0, ITEM_ID: 0, "timestamp": 99, LABEL: 1.0}])
    prepared.get_train_dataset().frame = pd.concat([train_frame, repeated_target], ignore_index=True)

    base_config = dict(
        sid_file=_sid_file(tmp_path),
        history_max_length=20,
        rl_add_item_alignment_tasks=False,
        rl_add_title_sequence_task=False,
    )
    codec = MiniOneRecSIDCodec.from_file(base_config["sid_file"], num_items=prepared.get_num_items())

    unfiltered = build_minionerec_rl_datasets(MiniOneRecConfig(**base_config), codec, prepared)
    filtered = build_minionerec_rl_datasets(MiniOneRecConfig(**base_config, rl_exclude_history=True), codec, prepared)

    unfiltered_rows = [unfiltered.train_dataset[index] for index in range(len(unfiltered.train_dataset))]
    filtered_rows = [filtered.train_dataset[index] for index in range(len(filtered.train_dataset))]
    repeated_sid = codec.item_sid(0) + "\n"

    assert any(row["completion"] == repeated_sid and 0 in row["excluded_item_ids"] for row in unfiltered_rows)
    assert not any(row["completion"] == repeated_sid and 0 in row["excluded_item_ids"] for row in filtered_rows)


def test_minionerec_rl_dataset_uses_active_source_alignment_tasks(tmp_path) -> None:
    prepared = _prepared_stub_dataset()
    config = MiniOneRecConfig(
        sid_file=_sid_file(tmp_path),
        rl_add_title_sequence_task=False,
    )
    codec = MiniOneRecSIDCodec.from_file(config.sid_file, num_items=prepared.get_num_items())

    rl_data = build_minionerec_rl_datasets(config, codec, prepared)
    prompts = [row["prompt"] for row in rl_data.train_dataset]

    assert "### User Input: \nWhich item has the title: Alpha Quest?\n\n### Response:\n" in prompts
    assert '### User Input: \nWhat is the title of item "<1_0><2_0><3_0>"?\n\n### Response:\n' not in prompts


def test_minionerec_rl_description_prompt_parses_source_list_literal(tmp_path) -> None:
    prepared = _prepared_stub_dataset()
    item_table = prepared.get_item_table()
    item_table.loc[item_table["item_id"] == 0, "description"] = "['first description', 'second description']"
    prepared._item_table = item_table
    config = MiniOneRecConfig(
        sid_file=_sid_file(tmp_path),
        rl_add_title_sequence_task=False,
    )
    codec = MiniOneRecSIDCodec.from_file(config.sid_file, num_items=prepared.get_num_items())

    rl_data = build_minionerec_rl_datasets(config, codec, prepared)
    prompts = [row["prompt"] for row in rl_data.train_dataset]

    assert (
        '### User Input: \nAn item can be described as follows: "first description". '
        "Which item is it describing?\n\n### Response:\n"
    ) in prompts


def test_minionerec_ranking_reward_matches_original_group_logic() -> None:
    config = MiniOneRecConfig(rl_num_generations=3, rl_reward_type="ranking")
    prompt2history = {"prompt": "history"}
    history2target = {"history": "<1_0><2_0><3_0>\n"}
    rule_reward, ndcg_reward = build_minionerec_reward_functions(
        config,
        prompt2history=prompt2history,
        history2target=history2target,
    )
    prompts = ["prompt", "prompt", "prompt"]
    completions = ["wrong\n", "<1_0><2_0><3_0>\n", "other\n"]

    assert rule_reward(prompts, completions) == [0.0, 1.0, 0.0]
    ranking = ndcg_reward(prompts, completions)
    assert ranking[1] == 0.0
    assert ranking[0] < ranking[2] < 0.0


def test_minionerec_ranking_only_reward_omits_rule_reward() -> None:
    config = MiniOneRecConfig(rl_num_generations=3, rl_reward_type="ranking_only")
    reward_funcs = build_minionerec_reward_functions(
        config,
        prompt2history={"prompt": "history"},
        history2target={"history": "<1_0><2_0><3_0>\n"},
    )

    assert len(reward_funcs) == 1
    rewards = reward_funcs[0](
        ["prompt", "prompt", "prompt"],
        ["wrong\n", "<1_0><2_0><3_0>\n", "other\n"],
    )

    assert rewards[1] == 0.0
    assert rewards[0] < rewards[2] < 0.0


def test_minionerec_reward_normalization_matches_source_mismatch() -> None:
    text = " <1_0><2_0><3_0> \n"

    assert normalize_minionerec_rule_text(text) == "<1_0><2_0><3_0>"
    assert normalize_minionerec_ranking_text(text) == " <1_0><2_0><3_0> "


def test_minionerec_grpo_defaults_follow_source_script() -> None:
    config = MiniOneRecConfig()

    assert config.rl_train_batch_size == 64
    assert config.rl_eval_batch_size == 128
    assert config.rl_gradient_accumulation_steps == 2
    assert config.rl_num_train_epochs == 2
    assert config.rl_learning_rate == 1.0e-5
    assert config.rl_beta == 1.0e-3
    assert config.rl_sync_ref_model is True
    assert config.rl_exclude_history is False
    assert config.rl_eval_steps == 0.0999
    assert config.allow_duplicate_sid_aliases is False


def test_minionerec_generation_selection_follows_recbole_topk_shape(tmp_path) -> None:
    config = MiniOneRecConfig(sid_file=_sid_file(tmp_path), topk=(4,))
    codec = MiniOneRecSIDCodec.from_file(config.sid_file, num_items=8)

    predictions, stats = _select_generated_item_ids(
        ["not-a-sid", "<1_2><2_2><3_2>", "<1_2><2_2><3_2>", "<1_3><2_3><3_3>"],
        codec,
        excluded=set(),
        k=4,
    )

    assert predictions == [2, 3, -1, -1]
    assert stats["invalid_generations"] == 1
    assert stats["duplicate_generations"] == 1
    assert stats["padded_predictions"] == 2


def test_minionerec_generation_collator_matches_eval_prompt_semantics(tmp_path) -> None:
    prepared = _prepared_stub_dataset()
    config = MiniOneRecConfig(sid_file=_sid_file(tmp_path), history_max_length=2)
    codec = MiniOneRecSIDCodec.from_file(config.sid_file, num_items=prepared.get_num_items())
    tokenizer = FakeTokenizer()
    collator = MiniOneRecGenerationCollator(config, prepared, tokenizer, codec)
    valid_frame = prepared.get_eval_dataset("valid").frame.iloc[[0]]

    batch = collator(valid_frame)
    valid_sft_example = build_sequence_sft_examples(config, codec, prepared, split="valid", eval_prompt=True)[0]
    expected_prompt = f'### User Input: \n{valid_sft_example["input"]}\n\n### Response:\n'

    assert tokenizer.calls[0] == MINIONEREC_SEQREC_INSTRUCTION
    assert expected_prompt in tokenizer.calls
    assert tuple(batch["input_ids"].shape) == (1, len(batch["attention_mask"][0]))


def test_minionerec_constraint_prefix_cache_reuses_excluded_sets(tmp_path) -> None:
    config = MiniOneRecConfig(
        sid_file=_sid_file(tmp_path),
        model_name_or_path="toy-model",
        constraint_cache_size=2,
    )
    codec = MiniOneRecSIDCodec.from_file(config.sid_file, num_items=8)
    eval_model = MiniOneRecGenerationRetrievalModel(config, object(), FakeTokenizer(), codec)

    first_entry, first_stats = eval_model._constraint_prefix_for_excluded(set())
    second_entry, second_stats = eval_model._constraint_prefix_for_excluded(set())
    third_entry, third_stats = eval_model._constraint_prefix_for_excluded({0})

    assert first_entry is second_entry
    assert third_entry is not first_entry
    assert first_stats == {"constraint_cache_misses": 1}
    assert second_stats == {"constraint_cache_hits": 1}
    assert third_stats == {"constraint_cache_misses": 1}


def test_minionerec_pipeline_smoke_uses_full_eval_and_writes_result(tmp_path, monkeypatch) -> None:
    ensure_stub_tables()
    sid_path = _sid_file(tmp_path)
    output_dir = tmp_path / "outputs"
    captured: dict[str, object] = {}

    def fake_run(self, task_data, *, output_dir):
        valid_frame = task_data.get_eval_dataset("valid").frame
        captured["stage"] = self.config.pipeline_stage
        captured["output_dir"] = Path(output_dir)
        captured["has_seen_item_ids"] = SEEN_ITEM_IDS in valid_frame.columns
        return {
            "fit": {"stage": self.config.pipeline_stage},
            "valid": {"metrics": {"recall@5": 0.0}},
            "test": {"metrics": {"recall@5": 0.0}},
            "checkpoint_path": Path(output_dir) / "checkpoint",
        }

    monkeypatch.setattr(MiniOneRecTrainer, "run", fake_run)
    cfg = OmegaConf.create(
        {
            "runtime": {"device": "cpu", "output_dir": str(output_dir)},
            "dataset": {
                "name": "stub_dataset",
                "processed_dir": str(tmp_path / "processed"),
                "split": {
                    "strategy": "leave_one_out",
                    "order": "chronological",
                    "per_user": True,
                    "valid_holdout_num": 1,
                    "test_holdout_num": 1,
                },
            },
            "model": {
                "name": "minionerec",
                "sid_file": sid_path,
                "model_name_or_path": "toy-model",
                "topk": [5],
                "metrics": ["recall"],
                "exclude_history": True,
            },
        }
    )

    result = run_experiment(cfg)

    assert result["fit"] == {"stage": "sft"}
    assert captured == {"stage": "sft", "output_dir": output_dir, "has_seen_item_ids": True}
    result_json = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result_json["checkpoint_path"] == str(output_dir / "checkpoint")
    assert MiniOneRecPipeline._build_eval_config(MiniOneRecConfig(topk=(5,), metrics=("recall",))).protocol == "full"
