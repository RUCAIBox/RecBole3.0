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
from recbole3.model.starec.candidates import (
    build_history_limited_frames,
    build_train_candidate_frame,
    select_history_eligible_user_ids,
)
from recbole3.model.starec.memory import STARecUserMemory
from recbole3.model.starec.model import STARecModel
from recbole3.model.starec.parser import complete_ranked_item_ids, parse_ranking_output, strip_think_blocks
from recbole3.model.starec.prompts import (
    build_memory_init_messages,
    build_ranking_messages,
    build_reflection_messages,
    resolve_item_domain,
)
from recbole3.model.starec.reward import compute_score, starec_ranking_reward
from recbole3.model.starec.training_data import (
    export_sft_from_teacher_trace,
    export_verl_ranking_from_teacher_trace,
    write_user_split_artifacts,
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
    assert config.train_init_interactions == 20
    assert config.history_min_length == 30
    assert config.selected_user_ids_path is None
    assert config.teacher_trace_path is None

    with pytest.raises(ValueError, match="api_batch"):
        STARecConfig(api_batch=0)
    with pytest.raises(ValueError, match="train_init_interactions"):
        STARecConfig(train_init_interactions=-1)
    with pytest.raises(ValueError, match="history_max_length"):
        STARecConfig(history_max_length=10, history_min_length=20)
    with pytest.raises(ValueError, match="item_domain"):
        STARecConfig(item_domain_singular="movie")


def test_starec_user_split_artifacts_are_fixed_count_and_disjoint(local_tmp_path: Path) -> None:
    result = write_user_split_artifacts(
        range(10),
        teacher_user_count=3,
        heldout_eval_user_count=2,
        seed=7,
        output_dir=local_tmp_path,
    )

    teacher_rows = [
        json.loads(line)
        for line in (local_tmp_path / "teacher_users.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    heldout_rows = [
        json.loads(line)
        for line in (local_tmp_path / "heldout_eval_users.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    teacher_users = {row["user_id"] for row in teacher_rows}
    heldout_users = {row["user_id"] for row in heldout_rows}

    assert result["teacher_user_count"] == 3
    assert result["heldout_eval_user_count"] == 2
    assert len(teacher_users) == 3
    assert len(heldout_users) == 2
    assert teacher_users.isdisjoint(heldout_users)


def test_starec_parser_requires_all_candidates_once() -> None:
    parsed = parse_ranking_output(
        "1. [ItemID: 2] Alpha\n2. [ItemID: 4] Bravo",
        [2, 4, 6],
    )

    assert not parsed.valid
    assert parsed.missing_item_ids == [6]
    assert complete_ranked_item_ids(parsed, [2, 4, 6]) == [2, 4, 6]


def test_starec_parser_ignores_think_blocks() -> None:
    output = "\n".join(
        [
            "<think>",
            "1. [ItemID: 6] Reasoning-only candidate mention",
            "</think>",
            "1. [ItemID: 2] Alpha",
            "2. [ItemID: 4] Bravo",
        ]
    )

    parsed = parse_ranking_output(output, [2, 4])

    assert "<think>" not in strip_think_blocks(output)
    assert parsed.valid
    assert parsed.ranked_item_ids == [2, 4]


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
                        message=types.SimpleNamespace(
                            content="Current User Description: SDK response",
                            reasoning_content="teacher reasoning trace",
                        )
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
    completion = model._complete_openai_with_reasoning([{"role": "user", "content": "ping"}])

    assert response == "Current User Description: SDK response"
    assert completion.content == "Current User Description: SDK response"
    assert completion.reasoning_content == "teacher reasoning trace"
    assert calls["client_kwargs"] == {
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com",
        "timeout": 60.0,
    }
    assert calls["create_kwargs"]["model"] == "deepseek-v4-flash"
    assert calls["create_kwargs"]["messages"] == [{"role": "user", "content": "ping"}]


def test_starec_item_text_template_and_feedback_score_field() -> None:
    model = STARecModel(
        STARecConfig(
            item_text_template="{title}. Artist/brand: {brand}",
            feedback_score_field="overall",
            feedback_positive_threshold=3,
        )
    )
    prepared_data = types.SimpleNamespace(
        config=types.SimpleNamespace(name="amazon2014_retrieval", category="CDs_and_Vinyl"),
        get_item_table=lambda: pd.DataFrame(
            [
                {ITEM_ID: 0, "title": "Blue Train", "brand": "John Coltrane", "metadata_text": "full metadata"},
                {ITEM_ID: 1, "title": "Kind of Blue", "brand": "", "metadata_text": "fallback metadata"},
            ]
        ),
        get_user_table=lambda: pd.DataFrame([{USER_ID: 0}]),
        get_num_items=lambda: 2,
    )

    model.prepare_metadata(prepared_data)

    assert model.item_text(0) == "Blue Train. Artist/brand: John Coltrane"
    assert model.item_text(1) == "Kind of Blue."
    assert model.record_feedback({"overall": 2.0, LABEL: None}) == "Actually Disliked"
    assert model.record_feedback({"overall": 4.0, LABEL: None}) == "Actually Liked"
    assert model.record_feedback({LABEL: 0.0}) == "Actually Disliked"


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


def test_starec_feedback_score_field_filters_targets() -> None:
    task_data = _FrameTaskData(
        train_frame=pd.DataFrame(
            [
                {USER_ID: 0, ITEM_ID: 0, LABEL: None, "overall": 5.0},
                {USER_ID: 0, ITEM_ID: 1, LABEL: None, "overall": 2.0},
                {USER_ID: 0, ITEM_ID: 2, LABEL: None, "overall": 4.0},
                {USER_ID: 0, ITEM_ID: 3, LABEL: None, "overall": 1.0},
                {USER_ID: 0, ITEM_ID: 4, LABEL: None, "overall": 5.0},
            ]
        ),
        valid_frame=pd.DataFrame([{USER_ID: 0, ITEM_ID: 5, LABEL: None, "overall": 4.0}]),
        test_frame=pd.DataFrame([{USER_ID: 0, ITEM_ID: 6, LABEL: None, "overall": 5.0}]),
        num_items=12,
    )
    config = STARecConfig(
        selected_user_count=1,
        train_init_interactions=2,
        history_min_length=5,
        backbone_topk=3,
        recall_budget=3,
        feedback_score_field="overall",
        feedback_positive_threshold=3,
    )

    train_frame, _, _, selected_user_ids = build_history_limited_frames(task_data, model_config=config)
    assert selected_user_ids == (0,)
    assert train_frame["overall"].tolist() == [5.0, 2.0, 4.0, 1.0, 5.0]

    task_data._train_dataset = FrameDataset(train_frame)
    train_candidate_frame = build_train_candidate_frame(task_data, model_config=config)

    negative_rows = train_candidate_frame.loc[train_candidate_frame["overall"] <= 3]
    positive_rows = train_candidate_frame.loc[train_candidate_frame["overall"] > 3]
    assert negative_rows[CANDIDATE_ITEM_IDS].tolist() == [(), ()]
    assert all(len(candidate_ids) == 3 for candidate_ids in positive_rows[CANDIDATE_ITEM_IDS].tolist())


def test_starec_train_random_candidates_reject_sample_without_full_item_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        num_items=10,
    )
    config = STARecConfig(backbone_topk=3, candidate_seed=7)

    def _fail_arange(*args, **kwargs):
        raise AssertionError("train random candidate sampling should not materialize all item ids")

    monkeypatch.setattr(
        starec_candidates,
        "np",
        types.SimpleNamespace(arange=_fail_arange, random=starec_candidates.np.random),
    )

    first_frame = build_train_candidate_frame(task_data, model_config=config)
    second_frame = build_train_candidate_frame(task_data, model_config=config)

    assert first_frame[CANDIDATE_ITEM_IDS].tolist() == second_frame[CANDIDATE_ITEM_IDS].tolist()
    assert first_frame.loc[1, CANDIDATE_ITEM_IDS] == ()
    for candidate_ids, seen_item_ids in zip(
        first_frame.loc[[0, 2], CANDIDATE_ITEM_IDS].tolist(),
        first_frame.loc[[0, 2], SEEN_ITEM_IDS].tolist(),
        strict=True,
    ):
        assert len(candidate_ids) == 3
        assert len(set(candidate_ids)) == 3
        assert set(candidate_ids).isdisjoint(seen_item_ids)


def test_starec_train_random_candidates_counts_insufficient_availability() -> None:
    task_data = _FrameTaskData(
        train_frame=pd.DataFrame(
            [
                {USER_ID: 0, ITEM_ID: 0, LABEL: 0.0},
                {USER_ID: 0, ITEM_ID: 1, LABEL: 1.0},
            ]
        ),
        valid_frame=pd.DataFrame([{USER_ID: 0, ITEM_ID: 0, LABEL: 1.0}]),
        test_frame=pd.DataFrame([{USER_ID: 0, ITEM_ID: 1, LABEL: 1.0}]),
        num_items=2,
    )

    with pytest.raises(ValueError, match="only has 1 unmasked items .* user 0, but backbone_topk=2"):
        build_train_candidate_frame(task_data, model_config=STARecConfig(backbone_topk=2))


def test_starec_lightweight_eligible_user_selection_avoids_frame_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def _fail_user_frame(*args, **kwargs):
        raise AssertionError("select_history_eligible_user_ids should not materialize per-user frames")

    monkeypatch.setattr(starec_candidates, "_user_frame", _fail_user_frame)

    selected_user_ids = select_history_eligible_user_ids(
        task_data,
        model_config=STARecConfig(
            selected_user_count=-1,
            train_init_interactions=2,
            history_min_length=5,
            history_max_length=6,
        ),
    )

    assert selected_user_ids == (0,)


def test_starec_history_selection_can_use_explicit_user_artifact(local_tmp_path: Path) -> None:
    task_data = _FrameTaskData(
        train_frame=pd.DataFrame(
            [
                {USER_ID: user_id, ITEM_ID: user_id * 5 + offset, LABEL: 1.0}
                for user_id in range(3)
                for offset in range(3)
            ]
        ),
        valid_frame=pd.DataFrame([{USER_ID: user_id, ITEM_ID: user_id * 5 + 3, LABEL: 1.0} for user_id in range(3)]),
        test_frame=pd.DataFrame([{USER_ID: user_id, ITEM_ID: user_id * 5 + 4, LABEL: 1.0} for user_id in range(3)]),
        num_items=15,
    )
    user_path = local_tmp_path / "teacher_users.jsonl"
    user_path.write_text('{"user_id": 2}\n{"user_id": 0}\n', encoding="utf-8")

    _, _, _, selected_user_ids = build_history_limited_frames(
        task_data,
        model_config=STARecConfig(
            selected_user_count=-1,
            selected_user_ids_path=str(user_path),
            train_init_interactions=1,
            history_min_length=5,
            history_max_length=5,
        ),
    )

    assert selected_user_ids == (2, 0)


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


def test_starec_teacher_trace_exports_sft_and_verl_data(local_tmp_path: Path) -> None:
    trace_path = local_tmp_path / "teacher_trace.jsonl"
    trace_records = [
        _trace_record("init_memory", "init", user_id=0, sequence_index=-1, description="likes jazz"),
        _ranking_trace_record("rank-0", user_id=0, sequence_index=0, target_rank=1),
        _trace_record(
            "reflection",
            "reflect-0",
            user_id=0,
            sequence_index=0,
            description="likes jazz and remastered albums",
            previous_description="likes jazz",
        ),
        _ranking_trace_record("rank-1", user_id=0, sequence_index=1, target_rank=5),
        _trace_record(
            "reflection",
            "reflect-1",
            user_id=0,
            sequence_index=1,
            description="likes jazz and remastered albums",
            previous_description="likes jazz and remastered albums",
        ),
        _ranking_trace_record("rank-2", user_id=0, sequence_index=2, target_rank=7),
    ]
    trace_path.write_text("\n".join(json.dumps(record) for record in trace_records) + "\n", encoding="utf-8")

    sft_path = local_tmp_path / "starec_sft.jsonl"
    rejected_path = local_tmp_path / "starec_sft_rejected.jsonl"
    dataset_info_path = local_tmp_path / "dataset_info.json"
    sft_result = export_sft_from_teacher_trace(
        trace_path,
        sft_path,
        rejected_path=rejected_path,
        dataset_info_path=dataset_info_path,
        rank_threshold=5,
    )

    sft_rows = [json.loads(line) for line in sft_path.read_text(encoding="utf-8").splitlines()]
    rejected_rows = [json.loads(line) for line in rejected_path.read_text(encoding="utf-8").splitlines()]
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))

    assert sft_result["accepted"] == 4
    assert {row["turn_type"] for row in sft_rows} == {"init_memory", "ranking", "reflection"}
    assert all(row["messages"][-1]["role"] == "assistant" for row in sft_rows)
    assert any(row["reason"] == "reflection_noop" for row in rejected_rows)
    assert any(row["reason"] == "target_rank>5" for row in rejected_rows)
    assert dataset_info["starec_sft"]["formatting"] == "sharegpt"
    assert dataset_info["starec_sft"]["columns"] == {"messages": "messages"}

    rl_path = local_tmp_path / "starec_rl.jsonl"
    rl_result = export_verl_ranking_from_teacher_trace(trace_path, rl_path)
    rl_rows = [json.loads(line) for line in rl_path.read_text(encoding="utf-8").splitlines()]

    assert rl_result["accepted"] == 3
    assert rl_result["rank_threshold"] is None
    assert all(row["data_source"] == "starec_ranking" for row in rl_rows)
    assert json.loads(rl_rows[0]["reward_model"]["ground_truth"])["target_item_id"] == 1

    thresholded_rl_path = local_tmp_path / "starec_rl_thresholded.jsonl"
    thresholded_rl_result = export_verl_ranking_from_teacher_trace(
        trace_path,
        thresholded_rl_path,
        rank_threshold=5,
    )
    assert thresholded_rl_result["accepted"] == 2
    assert thresholded_rl_result["rank_threshold"] == 5


def test_starec_sft_think_tags_requires_reasoning_content(local_tmp_path: Path) -> None:
    trace_path = local_tmp_path / "teacher_trace_reasoning.jsonl"
    trace_records = [
        _trace_record(
            "init_memory",
            "init",
            user_id=0,
            sequence_index=-1,
            description="likes jazz",
            reasoning_content="Build a concise profile from the user's history.",
        ),
        _ranking_trace_record(
            "rank-0",
            user_id=0,
            sequence_index=0,
            target_rank=1,
            reasoning_content="The target best matches the current memory.",
        ),
        _trace_record(
            "reflection",
            "reflect-0",
            user_id=0,
            sequence_index=0,
            description="likes jazz and remastered albums",
            previous_description="likes jazz",
            reasoning_content="The feedback adds a stable remaster preference.",
        ),
        _ranking_trace_record("rank-1", user_id=0, sequence_index=1, target_rank=1),
    ]
    trace_path.write_text("\n".join(json.dumps(record) for record in trace_records) + "\n", encoding="utf-8")

    output_path = local_tmp_path / "starec_sft_reasoning.jsonl"
    rejected_path = local_tmp_path / "starec_sft_reasoning_rejected.jsonl"
    result = export_sft_from_teacher_trace(
        trace_path,
        output_path,
        rejected_path=rejected_path,
        rank_threshold=5,
        sft_reasoning_mode="think-tags",
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    rejected_rows = [json.loads(line) for line in rejected_path.read_text(encoding="utf-8").splitlines()]

    assert result["accepted"] == 3
    assert result["sft_reasoning_mode"] == "think-tags"
    assert all(row["messages"][-1]["content"].startswith("<think>\n") for row in rows)
    assert all("\n</think>\n\n" in row["messages"][-1]["content"] for row in rows)
    assert any(row["reason"] == "reasoning_content_missing" for row in rejected_rows)


def test_starec_verl_reward_uses_ranking_bins() -> None:
    output = "\n".join(
        [
            "1. [ItemID: 2] Other",
            "2. [ItemID: 3] Other",
            "3. [ItemID: 1] Target",
        ]
    )
    ground_truth = json.dumps({"target_item_id": 1, "candidate_item_ids": [2, 3, 1], "topk": 20})

    assert starec_ranking_reward("1. [ItemID: 1] Target", target_item_id=1) == 1.0
    assert starec_ranking_reward(output, target_item_id=1, candidate_item_ids=[2, 3, 1]) == 0.5
    assert compute_score("starec_ranking", output, ground_truth) == 0.5
    with_think = "<think>\n1. [ItemID: 1] Reasoning mention\n</think>\n1. [ItemID: 2] Other\n2. [ItemID: 1] Target"
    assert starec_ranking_reward(with_think, target_item_id=1, candidate_item_ids=[2, 1]) == 0.5
    assert starec_ranking_reward(with_think, target_item_id=1) == 0.5
    assert starec_ranking_reward("1. [ItemID: 9] Other", target_item_id=1) == -1.0
    with pytest.raises(ValueError, match="target_item_id"):
        compute_score("starec_ranking", output, "{}")


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
                "  teacher_trace_path: teacher_trace.jsonl",
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
    teacher_trace_path = output_dir / "teacher_trace.jsonl"
    assert memory_path.exists()
    assert sample_log_path.exists()
    assert teacher_trace_path.exists()

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

    trace_rows = [json.loads(line) for line in teacher_trace_path.read_text(encoding="utf-8").splitlines()]
    assert {row["split"] for row in trace_rows} == {"train"}
    assert {row["turn_type"] for row in trace_rows} == {"init_memory", "ranking", "reflection"}
    assert all(row["messages"] for row in trace_rows)
    assert all("reasoning_content" in row for row in trace_rows)
    assert all(row["reasoning_content"] is None for row in trace_rows)


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


def _trace_record(
    turn_type: str,
    trace_id: str,
    *,
    user_id: int,
    sequence_index: int,
    description: str,
    previous_description: str | None = None,
    reasoning_content: str | None = None,
) -> dict:
    record = {
        "trace_id": trace_id,
        "turn_type": turn_type,
        "split": "train",
        "sequence_index": sequence_index,
        "user_id": user_id,
        "messages": [{"role": "user", "content": f"{turn_type} prompt"}],
        "raw_output": (
            f"Current User Description: {description}"
            if turn_type == "init_memory"
            else f"Updated User Description: {description}"
        ),
    }
    if reasoning_content is not None:
        record["reasoning_content"] = reasoning_content
    if turn_type == "init_memory":
        record["current_user_description"] = description
    else:
        record["updated_user_description"] = description
        record["previous_user_description"] = previous_description or ""
    return record


def _ranking_trace_record(
    trace_id: str,
    *,
    user_id: int,
    sequence_index: int,
    target_rank: int,
    reasoning_content: str | None = None,
) -> dict:
    candidate_item_ids = [1, 2, 3, 4, 5, 6, 7]
    ranked_item_ids = list(candidate_item_ids)
    ranked_item_ids.remove(1)
    ranked_item_ids.insert(target_rank - 1, 1)
    raw_output = "\n".join(
        f"{rank}. [ItemID: {item_id}] Item {item_id}" for rank, item_id in enumerate(ranked_item_ids, start=1)
    )
    record = {
        "trace_id": trace_id,
        "turn_type": "ranking",
        "split": "train",
        "sequence_index": sequence_index,
        "user_id": user_id,
        "target_item_id": 1,
        "candidate_item_ids": candidate_item_ids,
        "messages": [{"role": "user", "content": "ranking prompt"}],
        "raw_output": raw_output,
        "parsed_ranking": ranked_item_ids,
        "parse_valid": True,
        "target_rank": target_rank,
    }
    if reasoning_content is not None:
        record["reasoning_content"] = reasoning_content
    return record


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
