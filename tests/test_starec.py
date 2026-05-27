from __future__ import annotations

import json
import sys
import shutil
import tempfile
import types
from pathlib import Path

import pandas as pd
import pytest

import recbole3.model.starec.candidates as starec_candidates
from recbole3.dataset import CANDIDATE_ITEM_IDS, FrameDataset, ITEM_ID, LABEL, SEEN_ITEM_IDS, USER_ID
from recbole3.model import STARecConfig, get_model_spec
from recbole3.model.starec.candidates import build_history_limited_frames, build_train_candidate_frame
from recbole3.model.starec.memory import STARecUserMemory
from recbole3.model.starec.model import STARecModel
from recbole3.model.starec.parser import complete_ranked_item_ids, parse_ranking_output
from recbole3.model.starec.prompts import (
    build_memory_init_messages,
    build_ranking_messages,
    build_reflection_messages,
    resolve_item_domain,
)
from recbole3.model.starec.trainer import _STARecProgressBar
from recbole3.run import compose_config, run_experiment
from tests.test_helpers import ensure_stub_tables


@pytest.fixture
def local_tmp_path() -> Path:
    root = Path(__file__).resolve().parents[1] / ".pytest_tmp"
    root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="starec-", dir=root))
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_starec_registered_as_native_model() -> None:
    spec = get_model_spec("starec")

    assert spec.config_cls is STARecConfig
    assert spec.model_data_cls is not None


def test_starec_config_rejects_non_random_candidates() -> None:
    with pytest.raises(ValueError, match="candidate_source=random"):
        STARecConfig(candidate_source="bm25")


def test_starec_config_defaults_to_serial_dispatch() -> None:
    config = STARecConfig()

    assert config.api_batch == 1
    assert not config.async_dispatch
    assert config.train_init_interactions == 10
    assert config.history_min_length == 20

    with pytest.raises(ValueError, match="api_batch"):
        STARecConfig(api_batch=0)
    with pytest.raises(ValueError, match="train_init_interactions"):
        STARecConfig(train_init_interactions=-1)
    with pytest.raises(ValueError, match="history_max_length"):
        STARecConfig(history_max_length=10, history_min_length=20)
    with pytest.raises(ValueError, match="item_domain"):
        STARecConfig(item_domain_singular="movie")


def test_starec_parser_requires_all_candidates_once() -> None:
    parsed = parse_ranking_output(
        "1. [ItemID: 2] Alpha\n2. [ItemID: 4] Bravo",
        [2, 4, 6],
    )

    assert not parsed.valid
    assert parsed.missing_item_ids == [6]
    assert complete_ranked_item_ids(parsed, [2, 4, 6]) == [2, 4, 6]


def test_starec_item_domain_heuristic_and_override() -> None:
    assert resolve_item_domain(dataset_name="amazon2014_retrieval", category="CDs_and_Vinyl") == (
        "CD or music product",
        "CDs or music products",
    )
    assert resolve_item_domain(dataset_name="ml-1m", category=None) == ("movie", "movies")
    assert resolve_item_domain(dataset_name="stub_dataset", category=None) == ("item", "items")
    assert resolve_item_domain(
        dataset_name="stub_dataset",
        category=None,
        override_singular="board game",
        override_plural="board games",
    ) == ("board game", "board games")


def test_starec_prompts_are_structured_and_domain_aware() -> None:
    memory = STARecUserMemory(
        user_id=7,
        profile_text="User Profile:\n- User ID: 7",
        current_user_description="Prefers instrumental jazz and remastered collections.",
    )
    memory.append_interaction(item_id=11, item_text="Blue Room Sessions", feedback="liked")
    domain = {
        "item_domain_singular": "CD or music product",
        "item_domain_plural": "CDs or music products",
    }

    prompt_sets = [
        build_memory_init_messages(
            profile_text=memory.profile_text,
            history_lines=["- [ItemID: 11] Blue Room Sessions"],
            **domain,
        ),
        build_ranking_messages(
            memory=memory,
            candidate_lines=[
                "- [ItemID: 11] Blue Room Sessions",
                "- [ItemID: 12] Stadium Anthems",
            ],
            history_limit=10,
            **domain,
        ),
        build_reflection_messages(
            memory=memory,
            target_line="- [ItemID: 12] Stadium Anthems",
            system_prediction="Predicted Disliked",
            actual_feedback="Actually Liked",
            history_limit=10,
            **domain,
        ),
    ]

    for messages in prompt_sets:
        text = _messages_text(messages)
        for tag in ("role", "task", "context", "constraints", "output_format", "example"):
            assert f"<{tag}>" in text
            assert f"</{tag}>" in text
        assert "CDs or music products" in text

    ranking_text = _messages_text(prompt_sets[1])
    assert "Use only candidate items listed in <candidate_items>" in ranking_text
    assert "Preserve the exact [ItemID: ...]" in ranking_text
    assert "Return every candidate exactly once" in ranking_text

    reflection_text = _messages_text(prompt_sets[2])
    assert "Updated User Description must be <= 120 words" in reflection_text
    assert "Do not list or copy the full interaction history" in reflection_text
    assert "only when the target item and feedback provide clear evidence" in reflection_text


def test_starec_openai_backend_uses_sdk_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            calls["create_kwargs"] = kwargs
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="Current User Description: SDK response")
                    )
                ]
            )

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs
            self.chat = types.SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setenv("STAREC_TEST_API_KEY", "test-key")

    model = STARecModel(
        STARecConfig(
            backend="openai",
            api_base_url="https://api.deepseek.com",
            api_model_name="deepseek-v4-flash",
            api_key_env="STAREC_TEST_API_KEY",
        )
    )

    response = model._complete_openai([{"role": "user", "content": "ping"}])

    assert response == "Current User Description: SDK response"
    assert calls["client_kwargs"] == {
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com",
        "timeout": 60.0,
    }
    assert calls["create_kwargs"]["model"] == "deepseek-v4-flash"
    assert calls["create_kwargs"]["messages"] == [{"role": "user", "content": "ping"}]


def test_starec_progress_bar_tracks_rows_and_users(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, object]] = []

    class _FakeTqdm:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))

        def update(self, count):
            events.append(("update", count))

        def set_postfix_str(self, value):
            events.append(("postfix", value))

        def close(self):
            events.append(("close", None))

    monkeypatch.setitem(sys.modules, "tqdm.auto", types.SimpleNamespace(tqdm=_FakeTqdm))

    with _STARecProgressBar(desc="[starec:test]", total_rows=3, total_users=2) as progress:
        progress.update_rows(1)
        progress.update_user_done()
        progress.update_rows(2)
        progress.update_user_done()

    assert events[0] == ("init", {"total": 3, "desc": "[starec:test]", "leave": True})
    assert [event for event in events if event[0] == "update"] == [("update", 1), ("update", 2)]
    assert ("postfix", "users=2/2") in events
    assert events[-1] == ("close", None)


def test_starec_history_selection_keeps_negative_history_but_excludes_negative_targets() -> None:
    task_data = _FrameTaskData(
        train_frame=pd.DataFrame(
            [
                {USER_ID: 0, ITEM_ID: 0, LABEL: 1.0},
                {USER_ID: 0, ITEM_ID: 1, LABEL: 0.0},
                {USER_ID: 0, ITEM_ID: 2, LABEL: 1.0},
                {USER_ID: 0, ITEM_ID: 3, LABEL: 0.0},
                {USER_ID: 0, ITEM_ID: 4, LABEL: 1.0},
                {USER_ID: 1, ITEM_ID: 5, LABEL: 1.0},
                {USER_ID: 1, ITEM_ID: 6, LABEL: 1.0},
            ]
        ),
        valid_frame=pd.DataFrame([{USER_ID: 0, ITEM_ID: 7, LABEL: 1.0}, {USER_ID: 1, ITEM_ID: 8, LABEL: 1.0}]),
        test_frame=pd.DataFrame([{USER_ID: 0, ITEM_ID: 9, LABEL: 1.0}, {USER_ID: 1, ITEM_ID: 10, LABEL: 1.0}]),
        num_items=12,
    )
    config = STARecConfig(
        selected_user_count=1,
        train_init_interactions=2,
        history_min_length=5,
        history_max_length=6,
        backbone_topk=3,
        recall_budget=3,
        shuffle=False,
        refresh_candidate_cache=True,
    )

    train_frame, valid_frame, test_frame, selected_user_ids = build_history_limited_frames(
        task_data,
        model_config=config,
    )
    assert selected_user_ids == (0,)
    assert train_frame[LABEL].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert valid_frame[ITEM_ID].tolist() == [7]
    assert test_frame[ITEM_ID].tolist() == [9]
    assert valid_frame[SEEN_ITEM_IDS].tolist() == [(1, 2, 3, 4)]
    assert test_frame[SEEN_ITEM_IDS].tolist() == [(1, 2, 3, 4, 7)]

    task_data._train_dataset = FrameDataset(train_frame)
    train_candidate_frame = build_train_candidate_frame(task_data, model_config=config)

    negative_rows = train_candidate_frame.loc[train_candidate_frame[LABEL] == 0.0]
    positive_rows = train_candidate_frame.loc[train_candidate_frame[LABEL] > 0.0]
    assert all(candidate_ids == () for candidate_ids in negative_rows[CANDIDATE_ITEM_IDS].tolist())
    assert all(len(candidate_ids) == 3 for candidate_ids in positive_rows[CANDIDATE_ITEM_IDS].tolist())


def test_starec_history_selection_fails_when_requested_users_are_ineligible() -> None:
    task_data = _FrameTaskData(
        train_frame=pd.DataFrame(
            [
                {USER_ID: 0, ITEM_ID: 0, LABEL: 1.0},
                {USER_ID: 0, ITEM_ID: 1, LABEL: 0.0},
                {USER_ID: 0, ITEM_ID: 2, LABEL: 1.0},
            ]
        ),
        valid_frame=pd.DataFrame([{USER_ID: 0, ITEM_ID: 3, LABEL: 1.0}]),
        test_frame=pd.DataFrame([{USER_ID: 0, ITEM_ID: 4, LABEL: 1.0}]),
        num_items=6,
    )

    with pytest.raises(ValueError, match="eligible"):
        build_history_limited_frames(
            task_data,
            model_config=STARecConfig(
                selected_user_count=1,
                train_init_interactions=2,
                history_min_length=20,
                history_max_length=20,
            ),
        )


def test_starec_history_selection_stops_after_requested_eligible_users(monkeypatch: pytest.MonkeyPatch) -> None:
    user_count = 30
    task_data = _FrameTaskData(
        train_frame=pd.DataFrame(
            [
                {USER_ID: user_id, ITEM_ID: user_id * 5 + offset, LABEL: 1.0}
                for user_id in range(user_count)
                for offset in range(3)
            ]
        ),
        valid_frame=pd.DataFrame(
            [{USER_ID: user_id, ITEM_ID: user_id * 5 + 3, LABEL: 1.0} for user_id in range(user_count)]
        ),
        test_frame=pd.DataFrame(
            [{USER_ID: user_id, ITEM_ID: user_id * 5 + 4, LABEL: 1.0} for user_id in range(user_count)]
        ),
        num_items=user_count * 5,
    )
    history_calls = 0
    original_history_limiter = starec_candidates._history_limited_user_frames

    def _counting_history_limiter(**kwargs):
        nonlocal history_calls
        history_calls += 1
        return original_history_limiter(**kwargs)

    monkeypatch.setattr(starec_candidates, "_history_limited_user_frames", _counting_history_limiter)

    _, _, _, selected_user_ids = starec_candidates.build_history_limited_frames(
        task_data,
        model_config=STARecConfig(
            selected_user_count=2,
            train_init_interactions=1,
            history_min_length=5,
            history_max_length=5,
            candidate_seed=7,
        ),
    )

    assert len(selected_user_ids) == 2
    assert history_calls == 2


def test_starec_pipeline_end_to_end_with_deterministic_backend(local_tmp_path: Path) -> None:
    ensure_stub_tables()
    config_dir = local_tmp_path / "configs"
    (config_dir / "dataset").mkdir(parents=True)
    (config_dir / "model").mkdir(parents=True)

    output_dir = local_tmp_path / "outputs"
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "defaults:",
                "  - dataset: stub_dataset",
                "  - model: starec_test",
                "  - _self_",
                "runtime:",
                "  device: cpu",
                f"  output_dir: {output_dir.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "dataset" / "stub_dataset.yaml").write_text(
        "\n".join(
            [
                "name: stub_dataset",
                f"processed_dir: {(local_tmp_path / 'processed').as_posix()}",
                "split:",
                "  strategy: leave_one_out",
                "  order: chronological",
                "  per_user: true",
                "  valid_holdout_num: 1",
                "  test_holdout_num: 1",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "model" / "starec_test.yaml").write_text(
        "\n".join(
            [
                "# @package _global_",
                "",
                "model:",
                "  name: starec",
                "  backend: deterministic",
                "  candidate_source: random",
                "  backbone_topk: 4",
                "  recall_budget: 3",
                "  has_gt: true",
                "  fix_pos: -1",
                "  shuffle: false",
                "  selected_user_count: -1",
                f"  candidate_cache_dir: {(local_tmp_path / 'candidate_cache').as_posix()}",
                f"  candidate_file_dir: {(local_tmp_path / 'candidate_files').as_posix()}",
                "  refresh_candidate_cache: true",
                "  item_text_field: title",
                "  fallback_item_text_field: metadata_text",
                "  reflection_mode: always",
                "  prediction_liked_threshold: 1",
                "  api_batch: 2",
                "  async_dispatch: true",
                "  train_init_interactions: 1",
                "  history_min_length: 4",
                "  memory_save_path: starec_memories.jsonl",
                "  sample_log_path: starec_samples.jsonl",
                "trainer:",
                "  batch_size: 1",
                "  shuffle: false",
                "  max_epochs: 0",
                "  save_inference_results: true",
                "  eval:",
                "    protocol: full",
                "    neg_sampling_num: 0",
                "    candidate_seed: 7",
                "    metrics:",
                "      - name: recall",
                "        ks: [3]",
                "      - name: ndcg",
                "        ks: [3]",
            ]
        ),
        encoding="utf-8",
    )

    result = run_experiment(compose_config(config_dir=config_dir))

    assert "recall@3" in result["test"]["metrics"]
    assert "ndcg@3" in result["test"]["metrics"]
    memory_path = output_dir / "starec_memories.jsonl"
    sample_log_path = output_dir / "starec_samples.jsonl"
    assert memory_path.exists()
    assert sample_log_path.exists()

    memories = [json.loads(line) for line in memory_path.read_text(encoding="utf-8").splitlines()]
    assert len(memories) == 2
    assert all("Recent evidence" in record["current_user_description"] for record in memories)
    assert {record["user_id"]: [item["item_id"] for item in record["interaction_history"]] for record in memories} == {
        0: [0, 1, 2],
        1: [4, 5, 6],
    }
    assert all(len(record["reflection_history"]) == 2 for record in memories)

    sample_rows = [json.loads(line) for line in sample_log_path.read_text(encoding="utf-8").splitlines()]
    assert {row["split"] for row in sample_rows} == {"train", "valid", "test"}
    for split in ("train", "valid", "test"):
        split_indexes = [row["sequence_index"] for row in sample_rows if row["split"] == split]
        assert split_indexes == sorted(split_indexes)
    train_rows = [row for row in sample_rows if row["split"] == "train"]
    valid_rows = [row for row in sample_rows if row["split"] == "valid"]
    test_rows = [row for row in sample_rows if row["split"] == "test"]
    assert train_rows
    assert valid_rows
    assert test_rows
    assert all(row["reflection_triggered"] for row in train_rows)
    assert all(row["reflection_triggered"] for row in valid_rows)
    assert all(not row["reflection_triggered"] for row in test_rows)
    assert all("Recent evidence" in row["memory_before_ranking"]["current_user_description"] for row in test_rows)


def test_starec_can_load_prewarmed_memories(local_tmp_path: Path) -> None:
    ensure_stub_tables()
    config_dir = local_tmp_path / "configs"
    (config_dir / "dataset").mkdir(parents=True)
    (config_dir / "model").mkdir(parents=True)

    memory_load_path = local_tmp_path / "loaded_memories.jsonl"
    memory_load_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "user_id": 0,
                        "profile_text": "User Profile:\n- User ID: 0",
                        "current_user_description": "Loaded profile zero",
                        "interaction_history": [],
                        "reflection_history": [],
                    }
                ),
                json.dumps(
                    {
                        "user_id": 1,
                        "profile_text": "User Profile:\n- User ID: 1",
                        "current_user_description": "Loaded profile one",
                        "interaction_history": [],
                        "reflection_history": [],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    output_dir = local_tmp_path / "outputs"
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "defaults:",
                "  - dataset: stub_dataset",
                "  - model: starec_test",
                "  - _self_",
                "runtime:",
                "  device: cpu",
                f"  output_dir: {output_dir.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "dataset" / "stub_dataset.yaml").write_text(
        "\n".join(
            [
                "name: stub_dataset",
                f"processed_dir: {(local_tmp_path / 'processed').as_posix()}",
                "split:",
                "  strategy: leave_one_out",
                "  order: chronological",
                "  per_user: true",
                "  valid_holdout_num: 1",
                "  test_holdout_num: 1",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "model" / "starec_test.yaml").write_text(
        "\n".join(
            [
                "# @package _global_",
                "",
                "model:",
                "  name: starec",
                "  backend: deterministic",
                "  candidate_source: random",
                "  backbone_topk: 4",
                "  recall_budget: 3",
                "  has_gt: true",
                "  fix_pos: -1",
                "  shuffle: false",
                "  selected_user_count: -1",
                f"  candidate_cache_dir: {(local_tmp_path / 'candidate_cache').as_posix()}",
                f"  candidate_file_dir: {(local_tmp_path / 'candidate_files').as_posix()}",
                "  refresh_candidate_cache: true",
                "  reflection_mode: none",
                "  history_min_length: 4",
                f"  memory_load_path: {memory_load_path.as_posix()}",
                "  skip_warmup_when_memory_loaded: true",
                "  memory_save_path: saved_memories.jsonl",
                "  sample_log_path: samples.jsonl",
                "trainer:",
                "  eval:",
                "    protocol: full",
                "    metrics:",
                "      - name: recall",
                "        ks: [3]",
            ]
        ),
        encoding="utf-8",
    )

    result = run_experiment(compose_config(config_dir=config_dir))

    assert result["loaded_memory_count"] == 2
    sample_rows = [json.loads(line) for line in (output_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["split"] for row in sample_rows} == {"valid", "test"}
    assert all(row["split"] != "train" for row in sample_rows)
    assert sample_rows[0]["memory_before_ranking"]["current_user_description"].startswith("Loaded profile")


def _messages_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages)


class _FrameTaskData:
    def __init__(self, *, train_frame: pd.DataFrame, valid_frame: pd.DataFrame, test_frame: pd.DataFrame, num_items: int):
        self.config = types.SimpleNamespace(name="frame_task")
        self._train_dataset = FrameDataset(train_frame)
        self._valid_dataset = FrameDataset(valid_frame)
        self._test_dataset = FrameDataset(test_frame)
        self._num_items = int(num_items)

    def get_train_dataset(self):
        return self._train_dataset

    def get_eval_dataset(self, split: str):
        return self._valid_dataset if split == "valid" else self._test_dataset

    def get_num_items(self) -> int:
        return self._num_items
