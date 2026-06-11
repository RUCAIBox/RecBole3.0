from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from recbole3.dataset import ITEM_ID, LABEL, SEEN_ITEM_IDS, TIMESTAMP, USER_ID
from recbole3.evaluation import EvalConfig, MetricSpec
from recbole3.model import (
    HISTORY_ITEM_IDS,
    ETEGRecConfig,
    ETEGRecModel,
    ETEGRecModelDataset,
    ETEGRecTrainer,
    ETEGRecTrainerConfig,
    get_model_spec,
)
from recbole3.model.etegrec.data import ETEGRecEvalCollator, ETEGRecTrainCollator
from recbole3.model.etegrec.pretrain_rqvae import RQVAE as ETEGRecPretrainRQVAE
from recbole3.model.etegrec.pretrain_rqvae import ETEGRecRQVAEPretrainConfig
from recbole3.model.etegrec.pretrain_trainer import ETEGRecRQVAEPretrainTrainer, ETEGRecRQVAEPretrainTrainerConfig
from recbole3.run import compose_config
from recbole3.trainer import CheckpointConfig, EarlyStoppingConfig, Trainer, TrainerConfig
from recbole3.tools.generate_etegrec_cf_hstu import _resolve_output_path
from torch.utils.data import DataLoader, TensorDataset
from tests.test_helpers import StubDataset, StubDatasetConfig


def _full_eval_config() -> EvalConfig:
    return EvalConfig(protocol="full")


def _full_eval_config_with_metrics() -> EvalConfig:
    return EvalConfig(protocol="full", metrics=(MetricSpec(name="recall", ks=(5,)),), exclude_history=True)


def _rows(dataset, columns: list[str]) -> list[dict]:
    return dataset.frame.loc[:, columns].to_dict("records")


def _semantic_emb_file(tmp_path: Path, *, num_items: int = 8, dim: int = 4) -> Path:
    embeddings = np.arange(num_items * dim, dtype=np.float32).reshape(num_items, dim)
    path = tmp_path / "semantic_emb.npy"
    np.save(path, embeddings)
    return path


def _tiny_etegrec_config(tmp_path: Path) -> ETEGRecConfig:
    return ETEGRecConfig(
        history_max_length=2,
        semantic_emb_file=str(_semantic_emb_file(tmp_path, num_items=8, dim=4)),
        semantic_hidden_size=4,
        code_num=8,
        code_length=4,
        num_emb_list=(8, 8, 8),
        e_dim=4,
        layers=(4,),
        num_layers=1,
        num_decoder_layers=1,
        d_model=8,
        d_ff=16,
        num_heads=2,
        d_kv=4,
        dropout_rate=0.0,
    )


def _tiny_etegrec_model_and_batch(tmp_path: Path) -> tuple[ETEGRecModel, dict[str, torch.Tensor]]:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = _tiny_etegrec_config(tmp_path)
    etegrec_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=config)
    model = ETEGRecModel(config)
    collator = model.build_train_collator(etegrec_data)
    batch = collator(etegrec_data.get_train_dataset().frame.iloc[:2])
    return model, batch


class _DummyAccelerator:
    def __init__(self, *, is_main_process: bool = True, num_processes: int = 1) -> None:
        self.is_main_process = is_main_process
        self.num_processes = num_processes
        self.prepare_calls: list[tuple[Any, ...]] = []
        self.wait_calls = 0
        self.unwrap_calls: list[Any] = []

    def accumulate(self, model):
        del model
        return nullcontext()

    def prepare(self, *args):
        self.prepare_calls.append(args)
        if len(args) == 1:
            return args[0]
        return args

    def unwrap_model(self, model):
        self.unwrap_calls.append(model)
        return model

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        del parameters, max_norm

    def wait_for_everyone(self):
        self.wait_calls += 1


class _CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class _CallOnlyETEGRecModel(ETEGRecModel):
    def __init__(self, config: ETEGRecConfig, output: dict[str, torch.Tensor]):
        super().__init__(config)
        self.output = output
        self.calls: list[dict[str, Any]] = []

    def __call__(self, batch, *args, **kwargs):
        self.calls.append({"batch": batch, "args": args, "kwargs": kwargs})
        return self.output

    def forward(self, batch, *args, **kwargs):  # pragma: no cover - should not be reached by DDP-safe trainer paths.
        del batch, args, kwargs
        raise AssertionError("trainer step should call model(...) instead of model.forward(...)")


def test_etegrec_model_registration_and_config_defaults() -> None:
    model_spec = get_model_spec("etegrec")

    assert model_spec.config_cls is ETEGRecConfig
    assert model_spec.model_cls is ETEGRecModel
    assert model_spec.model_data_cls is ETEGRecModelDataset
    assert model_spec.trainer_cls is ETEGRecTrainer
    assert model_spec.trainer_config_cls is ETEGRecTrainerConfig

    cfg = compose_config(overrides=["model=etegrec"])
    assert cfg.model.name == "etegrec"
    assert cfg.model.history_max_length == 50
    assert cfg.model.num_beams == 20
    assert list(cfg.model.eval_topk) == [5, 10]
    assert cfg.trainer.monitor == "ndcg@10"
    assert cfg.trainer.eval_steps == 1
    assert ETEGRecTrainerConfig(eval=_full_eval_config()).eval_steps == 1


def test_etegrec_create_accelerator_preserves_base_settings_and_enables_find_unused_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeDistributedDataParallelKwargs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAccelerator:
        def __init__(self, **kwargs):
            captured["accelerator_kwargs"] = kwargs

    monkeypatch.setattr("accelerate.Accelerator", FakeAccelerator)
    monkeypatch.setattr("accelerate.utils.DistributedDataParallelKwargs", FakeDistributedDataParallelKwargs)
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            mixed_precision="fp16",
            gradient_accumulation_steps=3,
            eval=_full_eval_config(),
        )
    )

    trainer.create_accelerator()

    accelerator_kwargs = captured["accelerator_kwargs"]
    assert accelerator_kwargs["mixed_precision"] == "fp16"
    assert accelerator_kwargs["gradient_accumulation_steps"] == 3
    assert len(accelerator_kwargs["kwargs_handlers"]) == 1
    assert accelerator_kwargs["kwargs_handlers"][0].kwargs == {"find_unused_parameters": True}


def test_etegrec_model_dataset_uses_recbole_sequential_histories(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    etegrec_data = ETEGRecModelDataset.from_task_dataset(
        prepared,
        model_config=ETEGRecConfig(
            history_max_length=2,
            semantic_emb_file=str(_semantic_emb_file(tmp_path)),
            semantic_hidden_size=4,
        ),
    )

    assert _rows(etegrec_data.get_train_dataset(), [USER_ID, ITEM_ID, TIMESTAMP, LABEL, HISTORY_ITEM_IDS]) == [
        {USER_ID: 0, ITEM_ID: 0, TIMESTAMP: 1, LABEL: 1.0, HISTORY_ITEM_IDS: ()},
        {USER_ID: 0, ITEM_ID: 1, TIMESTAMP: 2, LABEL: 1.0, HISTORY_ITEM_IDS: (0,)},
        {USER_ID: 1, ITEM_ID: 4, TIMESTAMP: 1, LABEL: 1.0, HISTORY_ITEM_IDS: ()},
        {USER_ID: 1, ITEM_ID: 5, TIMESTAMP: 2, LABEL: 1.0, HISTORY_ITEM_IDS: (4,)},
    ]
    eval_columns = [USER_ID, ITEM_ID, TIMESTAMP, LABEL, SEEN_ITEM_IDS, HISTORY_ITEM_IDS]
    assert _rows(etegrec_data.get_eval_dataset("valid"), eval_columns) == [
        {USER_ID: 0, ITEM_ID: 2, TIMESTAMP: 3, LABEL: 1.0, SEEN_ITEM_IDS: (0, 1), HISTORY_ITEM_IDS: (0, 1)},
        {USER_ID: 1, ITEM_ID: 6, TIMESTAMP: 3, LABEL: 1.0, SEEN_ITEM_IDS: (4, 5), HISTORY_ITEM_IDS: (4, 5)},
    ]
    assert _rows(etegrec_data.get_eval_dataset("test"), eval_columns) == [
        {USER_ID: 0, ITEM_ID: 3, TIMESTAMP: 4, LABEL: 1.0, SEEN_ITEM_IDS: (0, 1, 2), HISTORY_ITEM_IDS: (1, 2)},
        {USER_ID: 1, ITEM_ID: 7, TIMESTAMP: 4, LABEL: 1.0, SEEN_ITEM_IDS: (4, 5, 6), HISTORY_ITEM_IDS: (5, 6)},
    ]
    assert etegrec_data.semantic_embeddings.shape == (8, 4)
    assert etegrec_data.semantic_embeddings.dtype == torch.float32
    assert etegrec_data.semantic_embeddings[0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_etegrec_model_dataset_resolves_bare_semantic_embedding_file_under_data_dir(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    prepared._parser = type("ParserWithDataDir", (), {"data_dir": tmp_path})()
    emb_file = _semantic_emb_file(tmp_path)

    etegrec_data = ETEGRecModelDataset.from_task_dataset(
        prepared,
        model_config=ETEGRecConfig(
            semantic_emb_file=emb_file.name,
            semantic_hidden_size=4,
        ),
    )

    assert etegrec_data.semantic_embeddings.shape == (8, 4)
    assert etegrec_data.semantic_embeddings[0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_generate_etegrec_cf_hstu_resolves_bare_output_file_under_data_dir(tmp_path: Path) -> None:
    assert _resolve_output_path("etegrec_hstu_emb_256.npy", tmp_path) == tmp_path / "etegrec_hstu_emb_256.npy"


def test_generate_etegrec_cf_hstu_keeps_explicit_output_paths(tmp_path: Path) -> None:
    relative_with_dir = Path("outputs/stage7/etegrec_hstu_emb_256.npy")
    absolute = tmp_path / "custom.npy"

    assert _resolve_output_path(str(relative_with_dir), tmp_path) == relative_with_dir
    assert _resolve_output_path(str(absolute), tmp_path) == absolute


def test_etegrec_model_dataset_requires_semantic_embedding_file() -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    with pytest.raises(ValueError, match="semantic_emb_file"):
        ETEGRecModelDataset.from_task_dataset(prepared, model_config=ETEGRecConfig(semantic_emb_file=""))


def test_etegrec_model_dataset_rejects_missing_semantic_embedding_file(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    with pytest.raises(FileNotFoundError, match="semantic_emb_file"):
        ETEGRecModelDataset.from_task_dataset(
            prepared,
            model_config=ETEGRecConfig(semantic_emb_file=str(tmp_path / "missing.npy")),
        )


def test_etegrec_model_dataset_rejects_semantic_embedding_row_mismatch(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    with pytest.raises(ValueError, match="Expected 8 rows"):
        ETEGRecModelDataset.from_task_dataset(
            prepared,
            model_config=ETEGRecConfig(
                semantic_emb_file=str(_semantic_emb_file(tmp_path, num_items=7, dim=4)),
                semantic_hidden_size=4,
            ),
        )


def test_etegrec_model_dataset_rejects_semantic_embedding_dim_mismatch(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())

    with pytest.raises(ValueError, match="semantic_hidden_size"):
        ETEGRecModelDataset.from_task_dataset(
            prepared,
            model_config=ETEGRecConfig(
                semantic_emb_file=str(_semantic_emb_file(tmp_path, num_items=8, dim=3)),
                semantic_hidden_size=4,
            ),
        )


def test_etegrec_model_dataset_rejects_non_finite_semantic_embeddings(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    embeddings = np.zeros((8, 4), dtype=np.float32)
    embeddings[0, 0] = np.nan
    path = tmp_path / "bad.npy"
    np.save(path, embeddings)

    with pytest.raises(ValueError, match="NaN or infinite"):
        ETEGRecModelDataset.from_task_dataset(
            prepared,
            model_config=ETEGRecConfig(semantic_emb_file=str(path), semantic_hidden_size=4),
        )


def test_etegrec_model_initializes_padded_semantic_embedding(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = ETEGRecConfig(
        semantic_emb_file=str(_semantic_emb_file(tmp_path, num_items=8, dim=4)),
        semantic_hidden_size=4,
    )
    etegrec_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=config)
    model = ETEGRecModel(config)

    model.build_train_collator(etegrec_data)

    assert model.semantic_embedding is not None
    assert model.semantic_embedding.num_embeddings == 9
    assert model.semantic_embedding.embedding_dim == 4
    assert model.semantic_embedding.padding_idx == 0
    assert model.semantic_embedding.weight.requires_grad is False
    assert model.semantic_embedding.weight[0].tolist() == [0.0, 0.0, 0.0, 0.0]
    torch.testing.assert_close(model.semantic_embedding.weight[1:].detach(), etegrec_data.semantic_embeddings)


def test_etegrec_tiny_forward_returns_finite_loss(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = ETEGRecConfig(
        history_max_length=2,
        semantic_emb_file=str(_semantic_emb_file(tmp_path, num_items=8, dim=4)),
        semantic_hidden_size=4,
        code_num=8,
        code_length=4,
        num_emb_list=(8, 8, 8),
        e_dim=4,
        layers=(4,),
        num_layers=1,
        num_decoder_layers=1,
        d_model=8,
        d_ff=16,
        num_heads=2,
        d_kv=4,
        dropout_rate=0.0,
    )
    etegrec_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=config)
    model = ETEGRecModel(config)
    collator = model.build_train_collator(etegrec_data)
    batch = collator(etegrec_data.get_train_dataset().frame.iloc[:2])

    outputs = model(batch)
    loss = model.compute_loss(batch, outputs)

    assert outputs["logits"].shape == (2, 4, 8)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_etegrec_forward_rqvae_mode_returns_rqvae_loss_parts(tmp_path: Path) -> None:
    model, batch = _tiny_etegrec_model_and_batch(tmp_path)

    parts = model(batch, mode="rqvae")

    assert set(parts) == {"loss", "recon_loss", "rq_loss"}
    assert parts["loss"].shape == ()
    assert torch.isfinite(parts["loss"])


def test_etegrec_forward_rqvae_mode_detaches_auxiliary_losses(tmp_path: Path) -> None:
    model, batch = _tiny_etegrec_model_and_batch(tmp_path)

    parts = model(batch, mode="rqvae")

    assert parts["loss"].requires_grad
    assert parts["recon_loss"].requires_grad is False
    assert parts["rq_loss"].requires_grad is False


def test_etegrec_forward_rec_loss_mode_returns_loss_only_graph(tmp_path: Path) -> None:
    model, batch = _tiny_etegrec_model_and_batch(tmp_path)

    parts = model(
        batch,
        mode="rec_loss",
        use_alignment=False,
        rec_code_loss=3.0,
        rec_kl_loss=5.0,
        rec_dec_cl_loss=7.0,
    )

    assert set(parts) == {"loss", "code_loss", "kl_loss", "dec_cl_loss"}
    assert parts["loss"].shape == ()
    assert torch.isfinite(parts["loss"])
    assert parts["loss"].requires_grad
    assert parts["code_loss"].requires_grad is False
    assert parts["kl_loss"].requires_grad is False
    assert parts["dec_cl_loss"].requires_grad is False
    assert "logits" not in parts
    assert "outputs" not in parts
    assert "labels" not in parts


def test_etegrec_forward_rec_loss_mode_with_alignment(tmp_path: Path) -> None:
    model, batch = _tiny_etegrec_model_and_batch(tmp_path)

    def fake_alignment_loss_parts(batch, outputs):
        del batch
        code_loss = outputs["loss"]
        return {
            "code_loss": code_loss,
            "kl_loss": code_loss * 0 + 2.0,
            "dec_cl_loss": code_loss * 0 + 3.0,
        }

    model.compute_alignment_loss_parts = fake_alignment_loss_parts  # type: ignore[method-assign]
    parts = model(
        batch,
        mode="rec_loss",
        use_alignment=True,
        rec_code_loss=5.0,
        rec_kl_loss=7.0,
        rec_dec_cl_loss=11.0,
    )

    expected = 5.0 * parts["code_loss"].item() + 7.0 * 2.0 + 11.0 * 3.0
    assert parts["loss"].item() == pytest.approx(expected)
    assert parts["loss"].requires_grad
    assert parts["code_loss"].requires_grad is False
    assert parts["kl_loss"].requires_grad is False
    assert parts["dec_cl_loss"].requires_grad is False


def test_etegrec_forward_tokenizer_loss_mode_without_alignment(tmp_path: Path) -> None:
    model, batch = _tiny_etegrec_model_and_batch(tmp_path)

    def forbidden_alignment(batch, outputs):
        del batch, outputs
        raise AssertionError("tokenizer_loss mode should not compute alignment when disabled")

    model.compute_alignment_loss_parts = forbidden_alignment  # type: ignore[method-assign]
    parts = model(
        batch,
        mode="tokenizer_loss",
        use_alignment=False,
        id_vq_loss=2.0,
        id_code_loss=3.0,
        id_kl_loss=5.0,
        id_dec_cl_loss=7.0,
    )

    assert set(parts) == {"loss", "vq_loss", "code_loss", "kl_loss", "dec_cl_loss", "recon_loss", "rq_loss"}
    assert parts["loss"].shape == ()
    assert torch.isfinite(parts["loss"])
    assert parts["loss"].requires_grad
    for key in ("vq_loss", "code_loss", "kl_loss", "dec_cl_loss", "recon_loss", "rq_loss"):
        assert parts[key].requires_grad is False


def test_etegrec_forward_tokenizer_loss_mode_with_alignment(tmp_path: Path) -> None:
    model, batch = _tiny_etegrec_model_and_batch(tmp_path)

    def fake_alignment_loss_parts(batch, outputs):
        del batch
        code_loss = outputs["loss"]
        return {
            "code_loss": code_loss,
            "kl_loss": code_loss * 0 + 2.0,
            "dec_cl_loss": code_loss * 0 + 3.0,
        }

    model.compute_alignment_loss_parts = fake_alignment_loss_parts  # type: ignore[method-assign]
    parts = model(
        batch,
        mode="tokenizer_loss",
        use_alignment=True,
        id_vq_loss=5.0,
        id_code_loss=7.0,
        id_kl_loss=11.0,
        id_dec_cl_loss=13.0,
    )

    expected = 5.0 * parts["vq_loss"].item() + 7.0 * parts["code_loss"].item() + 11.0 * 2.0 + 13.0 * 3.0
    assert parts["loss"].item() == pytest.approx(expected)
    assert parts["loss"].requires_grad
    for key in ("vq_loss", "code_loss", "kl_loss", "dec_cl_loss", "recon_loss", "rq_loss"):
        assert parts[key].requires_grad is False


def test_etegrec_forward_finetune_mode_returns_loss_only(tmp_path: Path) -> None:
    model, batch = _tiny_etegrec_model_and_batch(tmp_path)

    outputs = model(batch, mode="finetune")

    assert set(outputs) == {"loss"}
    assert outputs["loss"].shape == ()
    assert torch.isfinite(outputs["loss"])


def test_etegrec_finetune_mode_does_not_call_alignment_or_rqvae_loss(tmp_path: Path) -> None:
    model, batch = _tiny_etegrec_model_and_batch(tmp_path)

    def forbidden_alignment(batch, outputs):
        del batch, outputs
        raise AssertionError("finetune mode should not compute alignment loss")

    def forbidden_rqvae_loss(item_tokens):
        del item_tokens
        raise AssertionError("finetune mode should not compute RQVAE loss")

    model.compute_alignment_loss_parts = forbidden_alignment  # type: ignore[method-assign]
    model.compute_rqvae_loss = forbidden_rqvae_loss  # type: ignore[method-assign]

    outputs = model(batch, mode="finetune")

    assert set(outputs) == {"loss"}
    assert torch.isfinite(outputs["loss"])


def test_etegrec_forward_rejects_unknown_mode(tmp_path: Path) -> None:
    model, batch = _tiny_etegrec_model_and_batch(tmp_path)

    with pytest.raises(ValueError, match="forward mode"):
        model(batch, mode="bad")


def test_etegrec_pretrain_rqvae_forward_with_kmeans_init_is_finite() -> None:
    config = ETEGRecRQVAEPretrainConfig(
        num_emb_list=(4, 4, 4),
        e_dim=4,
        layers=(4,),
        kmeans_init=True,
        kmeans_iters=2,
    )
    model = ETEGRecPretrainRQVAE(config, in_dim=4)
    embeddings = torch.randn(8, 4)

    out, rq_loss, indices = model(embeddings)
    loss, recon_loss = model.compute_loss(out, rq_loss, xs=embeddings)

    assert out.shape == embeddings.shape
    assert indices.shape == (8, 3)
    assert torch.isfinite(loss)
    assert torch.isfinite(recon_loss)


def test_etegrec_pretrain_rqvae_checkpoint_loads_into_main_model(tmp_path: Path) -> None:
    pretrain_config = ETEGRecRQVAEPretrainConfig(
        num_emb_list=(8, 8, 8),
        e_dim=4,
        layers=(4,),
        kmeans_init=False,
    )
    embeddings = torch.randn(8, 4)
    dataloader = DataLoader(
        TensorDataset(embeddings),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda rows: torch.stack([row[0] for row in rows]),
    )
    pretrain_model = ETEGRecPretrainRQVAE(pretrain_config, in_dim=4)
    pretrain_trainer = ETEGRecRQVAEPretrainTrainer(
        ETEGRecRQVAEPretrainTrainerConfig(
            epochs=1,
            batch_size=4,
            eval_step=1,
            warmup_epochs=0,
            num_workers=0,
            device="cpu",
            save_limit=1,
        ),
        pretrain_model,
        output_dir=tmp_path / "rqvae_pretrain",
    )

    result = pretrain_trainer.fit(dataloader)
    checkpoint_path = Path(result["best_collision_path"])

    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    semantic_path = _semantic_emb_file(tmp_path, num_items=8, dim=4)
    main_config = ETEGRecConfig(
        history_max_length=2,
        semantic_emb_file=str(semantic_path),
        rqvae_path=str(checkpoint_path),
        semantic_hidden_size=4,
        code_num=8,
        code_length=4,
        num_emb_list=(8, 8, 8),
        e_dim=4,
        layers=(4,),
        num_layers=1,
        num_decoder_layers=1,
        d_model=8,
        d_ff=16,
        num_heads=2,
        d_kv=4,
        dropout_rate=0.0,
    )
    etegrec_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=main_config)
    main_model = ETEGRecModel(main_config)

    main_model.build_train_collator(etegrec_data)
    codes = main_model.refresh_item_codes()

    assert checkpoint_path.exists()
    assert codes.shape == (9, 4)
    assert int(codes[1:, :-1].min()) >= 0


def test_etegrec_alignment_losses_are_finite(tmp_path: Path) -> None:
    model, batch = _tiny_etegrec_model_and_batch(tmp_path)

    outputs = model(batch)
    parts = model.compute_alignment_loss_parts(batch, outputs)

    assert parts["code_loss"].shape == ()
    assert parts["kl_loss"].shape == ()
    assert parts["dec_cl_loss"].shape == ()
    assert torch.isfinite(parts["code_loss"])
    assert torch.isfinite(parts["kl_loss"])
    assert torch.isfinite(parts["dec_cl_loss"])


def test_etegrec_alignment_disabled_before_warm_epoch() -> None:
    model = ETEGRecModel(ETEGRecConfig())
    batch = {"targets": torch.tensor([[1], [2]])}
    outputs = {"loss": torch.tensor(2.0)}
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            warm_epoch=10,
            rec_code_loss=3.0,
            rec_kl_loss=5.0,
            rec_dec_cl_loss=7.0,
            eval=_full_eval_config(),
        )
    )

    def fake_alignment_loss_parts(batch, outputs):
        del batch, outputs
        raise AssertionError("alignment loss should not be computed before warm_epoch")

    model.compute_alignment_loss_parts = fake_alignment_loss_parts  # type: ignore[method-assign]
    parts = trainer._build_recommender_loss_parts(model, batch, outputs=outputs, epoch=1)

    assert parts["loss"].item() == pytest.approx(6.0)
    assert parts["kl_loss"].item() == pytest.approx(0.0)
    assert parts["dec_cl_loss"].item() == pytest.approx(0.0)


def test_etegrec_tokenizer_alignment_not_computed_before_warm_epoch() -> None:
    model = ETEGRecModel(ETEGRecConfig())
    batch = {"targets": torch.tensor([[1], [2]])}
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            warm_epoch=10,
            id_vq_loss=2.0,
            id_code_loss=4.0,
            id_kl_loss=5.0,
            id_dec_cl_loss=7.0,
            eval=_full_eval_config(),
        )
    )

    def fake_rqvae_loss(item_tokens):
        del item_tokens
        return {
            "loss": torch.tensor(11.0),
            "recon_loss": torch.tensor(3.0),
            "rq_loss": torch.tensor(5.0),
        }

    def fake_alignment_loss_parts(batch, outputs):
        del batch, outputs
        raise AssertionError("alignment loss should not be computed before warm_epoch")

    model.compute_rqvae_loss = fake_rqvae_loss  # type: ignore[method-assign]
    model.compute_alignment_loss_parts = fake_alignment_loss_parts  # type: ignore[method-assign]
    rqvae_parts = model.compute_rqvae_loss(batch["targets"])
    parts = trainer._build_tokenizer_loss_parts(model, batch, rqvae_parts, epoch=1)

    assert parts["loss"].item() == pytest.approx(2.0 * 11.0)
    assert parts["code_loss"].item() == pytest.approx(0.0)
    assert parts["kl_loss"].item() == pytest.approx(0.0)
    assert parts["dec_cl_loss"].item() == pytest.approx(0.0)


def test_etegrec_alignment_enabled_after_warm_epoch() -> None:
    model = ETEGRecModel(ETEGRecConfig())
    batch = {"targets": torch.tensor([[1], [2]])}
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            warm_epoch=0,
            rec_code_loss=3.0,
            rec_kl_loss=5.0,
            rec_dec_cl_loss=7.0,
            eval=_full_eval_config(),
        )
    )

    def fake_alignment_loss_parts(batch, outputs):
        del batch, outputs
        return {
            "code_loss": torch.tensor(2.0),
            "kl_loss": torch.tensor(11.0),
            "dec_cl_loss": torch.tensor(13.0),
        }

    model.compute_alignment_loss_parts = fake_alignment_loss_parts  # type: ignore[method-assign]
    outputs = {"loss": torch.tensor(2.0)}
    parts = trainer._build_recommender_loss_parts(model, batch, outputs=outputs, epoch=1)

    assert parts["loss"].item() == pytest.approx(3.0 * 2.0 + 5.0 * 11.0 + 7.0 * 13.0)
    assert parts["kl_loss"].item() == pytest.approx(11.0)
    assert parts["dec_cl_loss"].item() == pytest.approx(13.0)


def test_etegrec_id_step_includes_id_code_loss_after_warm_epoch() -> None:
    model = ETEGRecModel(ETEGRecConfig())
    batch = {"targets": torch.tensor([[1], [2]])}
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            warm_epoch=0,
            id_vq_loss=2.0,
            id_code_loss=4.0,
            id_kl_loss=0.0,
            id_dec_cl_loss=0.0,
            eval=_full_eval_config(),
        )
    )

    def fake_rqvae_loss(item_tokens):
        del item_tokens
        return {
            "loss": torch.tensor(11.0),
            "recon_loss": torch.tensor(3.0),
            "rq_loss": torch.tensor(5.0),
        }

    def fake_forward(batch, *, mode="rec"):
        del batch
        if mode == "rqvae":
            return fake_rqvae_loss(torch.tensor([[1], [2]]))
        return {}

    def fake_alignment_loss_parts(batch, outputs):
        del batch, outputs
        return {
            "code_loss": torch.tensor(7.0),
            "kl_loss": torch.tensor(13.0),
            "dec_cl_loss": torch.tensor(17.0),
        }

    model.compute_rqvae_loss = fake_rqvae_loss  # type: ignore[method-assign]
    model.forward = fake_forward  # type: ignore[method-assign]
    model.compute_alignment_loss_parts = fake_alignment_loss_parts  # type: ignore[method-assign]
    rqvae_parts = model(batch, mode="rqvae")
    parts = trainer._build_tokenizer_loss_parts(model, batch, rqvae_parts, epoch=1)

    assert parts["loss"].item() == pytest.approx(2.0 * 11.0 + 4.0 * 7.0)
    assert parts["code_loss"].item() == pytest.approx(7.0)


def test_etegrec_freeze_policy_keeps_only_active_side_trainable(tmp_path: Path) -> None:
    model, _ = _tiny_etegrec_model_and_batch(tmp_path)
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(eval=_full_eval_config()))

    trainer._set_train_mode_for_recommender_step(model)
    assert all(parameter.requires_grad for parameter in model.recommender_parameters())
    assert not any(parameter.requires_grad for parameter in model.rqvae_parameters())
    assert model.semantic_embedding is not None
    assert model.semantic_embedding.weight.requires_grad is False

    trainer._set_train_mode_for_tokenizer_step(model)
    assert not any(parameter.requires_grad for parameter in model.recommender_parameters())
    assert all(parameter.requires_grad for parameter in model.rqvae_parameters())
    assert model.semantic_embedding.weight.requires_grad is False


def test_etegrec_builds_rec_and_id_cosine_schedulers() -> None:
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            lr_scheduler_type="cosine",
            warmup_steps=8,
            max_epochs=3,
            cycle=2,
            eval=_full_eval_config(),
        )
    )
    dataloader = DataLoader(TensorDataset(torch.arange(5)), batch_size=1)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
    steps = trainer._resolve_scheduler_steps(dataloader)

    rec_scheduler = trainer._build_scheduler(
        optimizer,
        total_steps=steps["rec_total_steps"],
        warmup_steps=steps["rec_warmup_steps"],
    )
    id_scheduler = trainer._build_scheduler(
        optimizer,
        total_steps=steps["id_total_steps"],
        warmup_steps=steps["id_warmup_steps"],
    )

    assert rec_scheduler is not None
    assert id_scheduler is not None


def test_etegrec_scheduler_steps_only_active_optimizer() -> None:
    rec_param = torch.nn.Parameter(torch.tensor(1.0))
    id_param = torch.nn.Parameter(torch.tensor(1.0))
    model = _CallOnlyETEGRecModel(
        ETEGRecConfig(),
        {"loss": rec_param * 0 + torch.tensor(2.0)},
    )
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(warm_epoch=10, eval=_full_eval_config()))
    accelerator = _DummyAccelerator()
    rec_optimizer = torch.optim.SGD([rec_param], lr=1e-3)
    id_optimizer = torch.optim.SGD([id_param], lr=1e-3)
    rec_scheduler = _CountingScheduler()
    id_scheduler = _CountingScheduler()

    model.recommender_parameters = lambda: [rec_param]  # type: ignore[method-assign]
    model.rqvae_parameters = lambda: [id_param]  # type: ignore[method-assign]

    def fake_rqvae_loss(item_tokens):
        del item_tokens
        raise AssertionError("tokenizer step should call model(..., mode='tokenizer_loss')")

    model.compute_rqvae_loss = fake_rqvae_loss  # type: ignore[method-assign]
    batch = {"targets": torch.tensor([[1], [2]])}

    trainer._train_recommender_step(
        model,
        batch,
        rec_optimizer,
        rec_scheduler,
        accelerator,
        epoch=1,
    )
    assert rec_scheduler.steps == 1
    assert id_scheduler.steps == 0
    assert model.calls[-1]["kwargs"] == {
        "mode": "rec_loss",
        "use_alignment": False,
        "rec_code_loss": 1.0,
        "rec_kl_loss": 0.0,
        "rec_dec_cl_loss": 0.0,
    }

    id_param.requires_grad_(True)
    model.output = {
        "loss": id_param * 0 + torch.tensor(3.0),
        "vq_loss": torch.tensor(3.0),
        "code_loss": torch.tensor(0.0),
        "kl_loss": torch.tensor(0.0),
        "dec_cl_loss": torch.tensor(0.0),
        "recon_loss": torch.tensor(1.0),
        "rq_loss": torch.tensor(1.0),
    }
    trainer._train_tokenizer_step(
        model,
        batch,
        id_optimizer,
        id_scheduler,
        accelerator,
        epoch=1,
    )
    assert rec_scheduler.steps == 1
    assert id_scheduler.steps == 1
    assert model.calls[-1]["kwargs"] == {
        "mode": "tokenizer_loss",
        "use_alignment": False,
        "id_vq_loss": 1.0,
        "id_code_loss": 0.0,
        "id_kl_loss": 0.0,
        "id_dec_cl_loss": 0.0,
    }


def test_etegrec_tokenizer_step_does_not_call_unwrapped_alignment_helper() -> None:
    id_param = torch.nn.Parameter(torch.tensor(1.0))
    model = _CallOnlyETEGRecModel(
        ETEGRecConfig(),
        {
            "loss": id_param * 0 + torch.tensor(3.0),
            "vq_loss": torch.tensor(3.0),
            "code_loss": torch.tensor(7.0),
            "kl_loss": torch.tensor(11.0),
            "dec_cl_loss": torch.tensor(13.0),
        },
    )
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(warm_epoch=0, eval=_full_eval_config()))
    accelerator = _DummyAccelerator()
    optimizer = torch.optim.SGD([id_param], lr=1e-3)

    model.recommender_parameters = lambda: []  # type: ignore[method-assign]
    model.rqvae_parameters = lambda: [id_param]  # type: ignore[method-assign]

    def forbidden_alignment(batch, outputs):
        del batch, outputs
        raise AssertionError("tokenizer step should compute alignment inside model(..., mode='tokenizer_loss')")

    model.compute_alignment_loss_parts = forbidden_alignment  # type: ignore[method-assign]
    batch = {"targets": torch.tensor([[1], [2]])}

    trainer._train_tokenizer_step(model, batch, optimizer, None, accelerator, epoch=1)

    assert model.calls[-1]["kwargs"] == {
        "mode": "tokenizer_loss",
        "use_alignment": True,
        "id_vq_loss": 1.0,
        "id_code_loss": 0.0,
        "id_kl_loss": 0.0,
        "id_dec_cl_loss": 0.0,
    }


def test_etegrec_scheduler_uses_original_etegrec_step_estimate() -> None:
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            max_epochs=7,
            cycle=3,
            gradient_accumulation_steps=2,
            warmup_steps=8,
            eval=_full_eval_config(),
        )
    )
    dataloader = DataLoader(TensorDataset(torch.arange(10)), batch_size=1)

    steps = trainer._resolve_scheduler_steps(dataloader)

    assert trainer._get_num_update_steps_per_epoch(dataloader) == 5
    assert trainer._get_total_training_steps(dataloader) == 35
    assert steps["rec_total_steps"] == 35
    assert steps["id_total_steps"] == 35 // 3
    assert steps["rec_warmup_steps"] == 8
    assert steps["id_warmup_steps"] == 8 // 3


def test_etegrec_scheduler_disabled_by_default() -> None:
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(eval=_full_eval_config()))
    dataloader = DataLoader(TensorDataset(torch.arange(2)), batch_size=1)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
    steps = trainer._resolve_scheduler_steps(dataloader)

    rec_scheduler = trainer._build_scheduler(
        optimizer,
        total_steps=steps["rec_total_steps"],
        warmup_steps=steps["rec_warmup_steps"],
    )
    id_scheduler = trainer._build_scheduler(
        optimizer,
        total_steps=steps["id_total_steps"],
        warmup_steps=steps["id_warmup_steps"],
    )

    assert rec_scheduler is None
    assert id_scheduler is None


def test_etegrec_scheduler_tiny_total_steps_with_large_warmup() -> None:
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            lr_scheduler_type="cosine",
            warmup_steps=8000,
            max_epochs=1,
            cycle=4,
            eval=_full_eval_config(),
        )
    )
    dataloader = DataLoader(TensorDataset(torch.arange(1)), batch_size=1)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
    steps = trainer._resolve_scheduler_steps(dataloader)

    scheduler = trainer._build_scheduler(
        optimizer,
        total_steps=steps["id_total_steps"],
        warmup_steps=steps["id_warmup_steps"],
    )

    assert scheduler is not None


def test_etegrec_joint_schedulers_are_prepared_when_enabled() -> None:
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(lr_scheduler_type="cosine", warmup_steps=2, eval=_full_eval_config()))
    accelerator = _DummyAccelerator()
    model = ETEGRecModel(ETEGRecConfig())
    rec_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
    id_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
    train_dataloader = DataLoader(TensorDataset(torch.arange(2)), batch_size=1)
    rec_scheduler = trainer._build_scheduler(rec_optimizer, total_steps=2, warmup_steps=0)
    id_scheduler = trainer._build_scheduler(id_optimizer, total_steps=1, warmup_steps=0)

    prepared = trainer._prepare_joint_training_components(
        accelerator,
        model,
        rec_optimizer,
        id_optimizer,
        train_dataloader,
        rec_scheduler,
        id_scheduler,
    )

    assert prepared == (model, rec_optimizer, id_optimizer, train_dataloader, rec_scheduler, id_scheduler)
    assert accelerator.prepare_calls == [(model, rec_optimizer, id_optimizer, train_dataloader, rec_scheduler, id_scheduler)]
    assert None not in accelerator.prepare_calls[0]


def test_etegrec_scheduler_none_prepare_safe() -> None:
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(eval=_full_eval_config()))
    accelerator = _DummyAccelerator()
    model = ETEGRecModel(ETEGRecConfig())
    rec_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
    id_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
    train_dataloader = DataLoader(TensorDataset(torch.arange(2)), batch_size=1)

    prepared = trainer._prepare_joint_training_components(
        accelerator,
        model,
        rec_optimizer,
        id_optimizer,
        train_dataloader,
        None,
        None,
    )

    assert prepared == (model, rec_optimizer, id_optimizer, train_dataloader, None, None)
    assert accelerator.prepare_calls == [(model, rec_optimizer, id_optimizer, train_dataloader)]
    assert None not in accelerator.prepare_calls[0]


def test_etegrec_finetune_optimizer_scheduler_are_prepared() -> None:
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(eval=_full_eval_config()))
    accelerator = _DummyAccelerator()
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
    scheduler = _CountingScheduler()

    prepared_optimizer, prepared_scheduler = trainer._prepare_finetune_components(accelerator, optimizer, scheduler)

    assert prepared_optimizer is optimizer
    assert prepared_scheduler is scheduler
    assert accelerator.prepare_calls == [(optimizer, scheduler)]


def test_etegrec_finetune_optimizer_prepare_skips_none_scheduler() -> None:
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(eval=_full_eval_config()))
    accelerator = _DummyAccelerator()
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1e-3)

    prepared_optimizer, prepared_scheduler = trainer._prepare_finetune_components(accelerator, optimizer, None)

    assert prepared_optimizer is optimizer
    assert prepared_scheduler is None
    assert accelerator.prepare_calls == [(optimizer,)]


def test_etegrec_checkpoint_save_waits_for_everyone(tmp_path: Path) -> None:
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(eval=_full_eval_config()))
    accelerator = _DummyAccelerator()
    model = ETEGRecModel(ETEGRecConfig())
    saved_paths: list[Path] = []

    def fake_save_model_checkpoint(model, accelerator, path):
        del model, accelerator
        saved_paths.append(path)

    trainer._save_model_checkpoint = fake_save_model_checkpoint  # type: ignore[method-assign]
    checkpoint_path = tmp_path / "best_model.pt"

    trainer._save_model_checkpoint_and_wait(model, accelerator, checkpoint_path)

    assert saved_paths == [checkpoint_path]
    assert accelerator.wait_calls == 1


def test_etegrec_finetune_load_waits_before_loading_joint_best(tmp_path: Path) -> None:
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(eval=_full_eval_config()))
    accelerator = _DummyAccelerator()
    model = ETEGRecModel(ETEGRecConfig())
    events: list[str] = []

    def record_wait():
        events.append("wait")

    def fake_load_model_checkpoint(model, path):
        del model, path
        events.append("load")

    accelerator.wait_for_everyone = record_wait  # type: ignore[method-assign]
    trainer._load_model_checkpoint = fake_load_model_checkpoint  # type: ignore[method-assign]

    trainer._load_model_checkpoint_with_barrier(model, tmp_path / "best_model.pt", accelerator)

    assert events == ["wait", "load", "wait"]


def test_etegrec_single_process_evaluation_uses_super(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(eval=_full_eval_config()))
    model = ETEGRecModel(ETEGRecConfig())
    accelerator = _DummyAccelerator(num_processes=1)
    prepared_data = object()
    expected = {"split": "valid", "metrics": {"recall@5": 1.0}}
    calls: list[tuple[Any, Any, Any, bool]] = []

    def fake_super_run(self, model_arg, prepared_data_arg, split, accelerator, model_is_prepared):
        del self
        calls.append((model_arg, prepared_data_arg, accelerator, model_is_prepared))
        return expected

    monkeypatch.setattr(Trainer, "_run_evaluation", fake_super_run)

    result = trainer._run_evaluation(model, prepared_data, split="valid", accelerator=accelerator, model_is_prepared=True)

    assert result is expected
    assert calls == [(model, prepared_data, accelerator, True)]


def test_etegrec_multigpu_main_process_runs_full_evaluation_with_unwrapped_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(eval=_full_eval_config()))
    wrapped_model = object()
    eval_model = object()
    prepared_data = object()
    accelerator = _DummyAccelerator(is_main_process=True, num_processes=2)
    expected = {"split": "valid", "metrics": {"ndcg@10": 0.5}}
    calls: list[Any] = []

    accelerator.unwrap_model = lambda model: eval_model  # type: ignore[method-assign]

    def fake_full_eval(model, prepared_data, *, split):
        calls.append((model, prepared_data, split))
        return expected

    monkeypatch.setattr(trainer, "_run_unsharded_evaluation_on_main_process", fake_full_eval)
    monkeypatch.setattr(trainer, "_broadcast_evaluation_result", lambda result, accelerator: result)

    result = trainer._run_evaluation(wrapped_model, prepared_data, split="valid", accelerator=accelerator, model_is_prepared=True)

    assert result is expected
    assert calls == [(eval_model, prepared_data, "valid")]
    assert accelerator.wait_calls == 2


def test_etegrec_multigpu_non_main_process_skips_forward_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(eval=_full_eval_config()))
    accelerator = _DummyAccelerator(is_main_process=False, num_processes=2)
    expected = {"split": "valid", "metrics": {"ndcg@10": 0.5}}

    def forbidden_full_eval(*args, **kwargs):
        del args, kwargs
        raise AssertionError("non-main rank should not run full evaluation")

    def fake_broadcast(result, accelerator):
        assert result is None
        return expected

    monkeypatch.setattr(trainer, "_run_unsharded_evaluation_on_main_process", forbidden_full_eval)
    monkeypatch.setattr(trainer, "_broadcast_evaluation_result", fake_broadcast)

    result = trainer._run_evaluation(object(), object(), split="valid", accelerator=accelerator, model_is_prepared=True)

    assert result is expected
    assert accelerator.wait_calls == 2


def test_etegrec_broadcast_evaluation_result_returns_same_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    result = {"split": "valid", "metrics": {"recall@5": 0.25}}
    accelerator = _DummyAccelerator(is_main_process=True, num_processes=2)

    def fake_broadcast_object_list(object_list, src):
        assert src == 0
        assert object_list == [result]

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", fake_broadcast_object_list)

    assert ETEGRecTrainer._broadcast_evaluation_result(result, accelerator) is result


def test_etegrec_broadcast_evaluation_result_populates_non_main_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    result = {"split": "valid", "metrics": {"recall@5": 0.25}}
    accelerator = _DummyAccelerator(is_main_process=False, num_processes=2)

    def fake_broadcast_object_list(object_list, src):
        assert src == 0
        assert object_list == [None]
        object_list[0] = result

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", fake_broadcast_object_list)

    assert ETEGRecTrainer._broadcast_evaluation_result(None, accelerator) is result


def test_etegrec_eval_batch_size_still_applies_under_main_process_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            batch_size=8,
            eval_batch_size=2,
            eval=_full_eval_config(),
        )
    )
    seen_batch_sizes: list[int] = []

    class FakeMethod:
        protocol = "full"

        def build_eval_collate_fn(self, model, prepared_data):
            del model, prepared_data
            return lambda rows: ({"x": torch.stack([row[0] for row in rows])}, rows)

        def compute_metrics(self, batch_eval_data):
            del batch_eval_data
            return {}

    class FakePreparedData:
        def get_eval_dataset(self, split):
            del split
            return TensorDataset(torch.arange(4))

        def get_num_users(self):
            return 1

        def get_num_items(self):
            return 4

    monkeypatch.setattr(trainer, "create_evaluation_method", lambda prepared_data: FakeMethod())
    monkeypatch.setattr(
        trainer,
        "_collect_eval_batch",
        lambda method, model, model_inputs, records: seen_batch_sizes.append(len(records)) or object(),
    )

    result = trainer._run_unsharded_evaluation_on_main_process(ETEGRecModel(ETEGRecConfig()), FakePreparedData(), split="valid")

    assert result["split"] == "valid"
    assert seen_batch_sizes == [2, 2]


def test_etegrec_main_process_eval_moves_model_inputs_to_model_device(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(batch_size=2, eval_batch_size=2, eval=_full_eval_config()))
    seen: list[tuple[Any, torch.device]] = []

    class FakeMethod:
        protocol = "full"

        def build_eval_collate_fn(self, model, prepared_data):
            del model, prepared_data
            return lambda rows: ({"x": torch.stack([row[0] for row in rows])}, rows)

        def compute_metrics(self, batch_eval_data):
            del batch_eval_data
            return {"recall@5": 1.0}

    class FakePreparedData:
        def get_eval_dataset(self, split):
            del split
            return TensorDataset(torch.arange(2))

        def get_num_users(self):
            return 1

        def get_num_items(self):
            return 2

    model = torch.nn.Linear(1, 1)
    expected_device = next(model.parameters()).device
    original_move = trainer._move_to_device

    def record_move(value, device):
        seen.append((value, device))
        return original_move(value, device)

    monkeypatch.setattr(trainer, "create_evaluation_method", lambda prepared_data: FakeMethod())
    monkeypatch.setattr(trainer, "_move_to_device", record_move)

    def collect_batch(method, model, model_inputs, records):
        del method, model, records
        assert model_inputs["x"].device == expected_device
        return object()

    monkeypatch.setattr(trainer, "_collect_eval_batch", collect_batch)

    result = trainer._run_unsharded_evaluation_on_main_process(model, FakePreparedData(), split="valid")

    assert result["metrics"] == {"recall@5": 1.0}
    assert seen
    assert seen[0][1] == expected_device


def test_etegrec_move_to_device_handles_nested_inputs() -> None:
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(eval=_full_eval_config()))
    tensor = torch.tensor([1])
    value = {
        "tensor": tensor,
        "tuple": (tensor,),
        "list": [tensor],
        "plain": object(),
    }

    moved = trainer._move_to_device(value, torch.device("cpu"))

    assert moved["tensor"].device.type == "cpu"
    assert moved["tuple"][0].device.type == "cpu"
    assert moved["list"][0].device.type == "cpu"
    assert moved["plain"] is value["plain"]


def test_etegrec_multigpu_main_eval_does_not_prepare_dataloader(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(batch_size=2, eval_batch_size=1, eval=_full_eval_config()))
    accelerator = _DummyAccelerator(is_main_process=True, num_processes=2)

    class FakeMethod:
        protocol = "full"

        def build_eval_collate_fn(self, model, prepared_data):
            del model, prepared_data
            return lambda rows: ({"x": torch.stack([row[0] for row in rows])}, rows)

        def compute_metrics(self, batch_eval_data):
            del batch_eval_data
            return {}

    class FakePreparedData:
        def get_eval_dataset(self, split):
            del split
            return TensorDataset(torch.arange(2))

        def get_num_users(self):
            return 1

        def get_num_items(self):
            return 2

    monkeypatch.setattr(trainer, "create_evaluation_method", lambda prepared_data: FakeMethod())
    monkeypatch.setattr(trainer, "_collect_eval_batch", lambda method, model, model_inputs, records: object())
    monkeypatch.setattr(trainer, "_broadcast_evaluation_result", lambda result, accelerator: result)

    trainer._run_evaluation(torch.nn.Linear(1, 1), FakePreparedData(), split="valid", accelerator=accelerator, model_is_prepared=True)

    assert accelerator.prepare_calls == []


def test_etegrec_eval_batch_size_defaults_to_train_batch_size() -> None:
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            batch_size=8,
            eval_batch_size=None,
            eval=_full_eval_config(),
        )
    )
    dataset = TensorDataset(torch.arange(16))

    train_loader = trainer.build_dataloader(dataset, lambda rows: rows, shuffle=False)
    trainer._building_eval_dataloader = True
    try:
        eval_loader = trainer.build_dataloader(dataset, lambda rows: rows, shuffle=False)
    finally:
        trainer._building_eval_dataloader = False

    assert train_loader.batch_size == 8
    assert eval_loader.batch_size == 8


def test_etegrec_eval_batch_size_overrides_eval_only() -> None:
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            batch_size=8,
            eval_batch_size=2,
            eval=_full_eval_config(),
        )
    )
    dataset = TensorDataset(torch.arange(16))

    train_loader = trainer.build_dataloader(dataset, lambda rows: rows, shuffle=False)
    trainer._building_eval_dataloader = True
    try:
        eval_loader = trainer.build_dataloader(dataset, lambda rows: rows, shuffle=False)
    finally:
        trainer._building_eval_dataloader = False

    assert train_loader.batch_size == 8
    assert eval_loader.batch_size == 2
    assert trainer.config.batch_size == 8


def test_etegrec_default_eval_batch_size_matches_original() -> None:
    cfg = compose_config(overrides=["model=etegrec"])

    assert int(cfg.trainer.eval_batch_size) == 32


def test_etegrec_eval_batch_size_rejects_non_positive() -> None:
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            batch_size=8,
            eval_batch_size=0,
            eval=_full_eval_config(),
        )
    )
    trainer._building_eval_dataloader = True
    try:
        with pytest.raises(ValueError, match="eval_batch_size"):
            trainer.build_dataloader(TensorDataset(torch.arange(4)), lambda rows: rows, shuffle=False)
    finally:
        trainer._building_eval_dataloader = False


def test_etegrec_finetune_disabled_preserves_existing_fit_flow(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = _tiny_etegrec_config(tmp_path)
    etegrec_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=config)
    model = ETEGRecModel(config)
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            batch_size=2,
            shuffle=False,
            max_epochs=1,
            finetune_enabled=False,
            monitor="recall@5",
            checkpoint=CheckpointConfig(save_best=True, save_last=True),
            eval=_full_eval_config_with_metrics(),
        )
    )

    result = trainer.fit(model, etegrec_data, output_dir=tmp_path / "disabled")

    assert "finetune_train_history" not in result
    assert result["checkpoint_paths"]["best"].endswith("best_model.pt")
    assert Path(result["checkpoint_paths"]["best"]).exists()


def test_etegrec_finetune_requires_joint_best_checkpoint(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = _tiny_etegrec_config(tmp_path)
    etegrec_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=config)
    model = ETEGRecModel(config)
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            batch_size=2,
            shuffle=False,
            max_epochs=1,
            finetune_enabled=True,
            finetune_epochs=1,
            monitor="recall@5",
            checkpoint=CheckpointConfig(save_best=False, save_last=False),
            eval=_full_eval_config_with_metrics(),
        )
    )

    with pytest.raises(ValueError, match="joint best checkpoint"):
        trainer.fit(model, etegrec_data, output_dir=tmp_path / "missing-best")


def test_etegrec_finetune_loads_joint_best_before_training(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = _tiny_etegrec_config(tmp_path)
    etegrec_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=config)
    model = ETEGRecModel(config)
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            batch_size=2,
            shuffle=False,
            max_epochs=1,
            finetune_enabled=True,
            finetune_epochs=1,
            monitor="recall@5",
            checkpoint=CheckpointConfig(save_best=True, save_last=True),
            eval=_full_eval_config_with_metrics(),
        )
    )
    loaded_paths: list[Path] = []

    def fake_load_model_checkpoint(model, path):
        del model
        loaded_paths.append(path)

    trainer._load_model_checkpoint = fake_load_model_checkpoint  # type: ignore[method-assign]
    trainer.fit(model, etegrec_data, output_dir=tmp_path / "loads-best")

    assert loaded_paths
    assert loaded_paths[0].name == "best_model.pt"


def test_etegrec_finetune_uses_code_loss_only() -> None:
    rec_param = torch.nn.Parameter(torch.tensor(1.0))
    model = _CallOnlyETEGRecModel(
        ETEGRecConfig(),
        {"loss": rec_param * 0 + torch.tensor(2.0)},
    )
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(eval=_full_eval_config()))
    accelerator = _DummyAccelerator()
    optimizer = torch.optim.SGD([rec_param], lr=1e-3)
    scheduler = _CountingScheduler()

    model.recommender_parameters = lambda: [rec_param]  # type: ignore[method-assign]
    model.rqvae_parameters = lambda: []  # type: ignore[method-assign]

    def forbidden_alignment(batch, outputs):
        del batch, outputs
        raise AssertionError("finetune should not compute alignment loss")

    def forbidden_rqvae_loss(item_tokens):
        del item_tokens
        raise AssertionError("finetune should not compute RQVAE loss")

    def forbidden_compute_loss(batch, outputs):
        del batch, outputs
        raise AssertionError("finetune step should use model(..., mode='finetune') loss directly")

    model.compute_alignment_loss_parts = forbidden_alignment  # type: ignore[method-assign]
    model.compute_rqvae_loss = forbidden_rqvae_loss  # type: ignore[method-assign]
    model.compute_loss = forbidden_compute_loss  # type: ignore[method-assign]

    loss = trainer._finetune_step(
        model,
        {"targets": torch.tensor([[1], [2]])},
        optimizer,
        scheduler,
        accelerator,
    )

    assert loss.item() == pytest.approx(2.0)
    assert scheduler.steps == 1
    assert model.calls[-1]["kwargs"] == {"mode": "finetune"}


def test_etegrec_finetune_freezes_tokenizer_and_semantic_embedding(tmp_path: Path) -> None:
    model, batch = _tiny_etegrec_model_and_batch(tmp_path)
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(eval=_full_eval_config()))
    accelerator = _DummyAccelerator()
    optimizer = torch.optim.SGD(list(model.recommender_parameters()), lr=1e-3)

    trainer._finetune_step(model, batch, optimizer, None, accelerator)

    assert all(parameter.requires_grad for parameter in model.recommender_parameters())
    assert not any(parameter.requires_grad for parameter in model.rqvae_parameters())
    assert model.semantic_embedding is not None
    assert model.semantic_embedding.weight.requires_grad is False


def test_etegrec_finetune_saves_separate_best_checkpoint(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = _tiny_etegrec_config(tmp_path)
    etegrec_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=config)
    model = ETEGRecModel(config)
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            batch_size=2,
            shuffle=False,
            max_epochs=1,
            finetune_enabled=True,
            finetune_epochs=1,
            monitor="recall@5",
            checkpoint=CheckpointConfig(save_best=True, save_last=True),
            eval=_full_eval_config_with_metrics(),
        )
    )

    result = trainer.fit(model, etegrec_data, output_dir=tmp_path / "finetune")

    finetune_best = Path(result["finetune_checkpoint_paths"]["best"])
    assert finetune_best.name == "finetune_best_model.pt"
    assert finetune_best.exists()


def test_etegrec_finetune_updates_final_best_path_for_test(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = _tiny_etegrec_config(tmp_path)
    etegrec_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=config)
    model = ETEGRecModel(config)
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            batch_size=2,
            shuffle=False,
            max_epochs=1,
            finetune_enabled=True,
            finetune_epochs=1,
            monitor="recall@5",
            checkpoint=CheckpointConfig(save_best=True, save_last=True),
            eval=_full_eval_config_with_metrics(),
        )
    )

    result = trainer.fit(model, etegrec_data, output_dir=tmp_path / "final-best")

    assert result["checkpoint_paths"]["best"].endswith("finetune_best_model.pt")
    assert result["joint_checkpoint_paths"]["best"].endswith("best_model.pt")
    assert Path(result["checkpoint_paths"]["best"]).exists()
    assert Path(result["joint_checkpoint_paths"]["best"]).exists()


def test_etegrec_finetune_scheduler_defaults_to_cosine_zero_warmup() -> None:
    trainer = ETEGRecTrainer(ETEGRecTrainerConfig(finetune_enabled=True, eval=_full_eval_config()))
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1e-3)

    scheduler = trainer._build_scheduler(
        optimizer,
        total_steps=2,
        warmup_steps=int(trainer.config.finetune_warmup_steps),
        scheduler_type=trainer.config.finetune_lr_scheduler_type,
        use_config_scheduler=False,
    )

    assert trainer.config.finetune_lr_scheduler_type == "cosine"
    assert trainer.config.finetune_warmup_steps == 0
    assert scheduler is not None
    optimizer.step()
    scheduler.step()


def test_etegrec_rqvae_checkpoint_load_happens_after_module_initialization(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    base_config = ETEGRecConfig(
        history_max_length=2,
        semantic_emb_file=str(_semantic_emb_file(tmp_path, num_items=8, dim=4)),
        semantic_hidden_size=4,
        code_num=8,
        code_length=4,
        num_emb_list=(8, 8, 8),
        e_dim=4,
        layers=(4,),
        num_layers=1,
        num_decoder_layers=1,
        d_model=8,
        d_ff=16,
        num_heads=2,
        d_kv=4,
        dropout_rate=0.0,
    )
    etegrec_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=base_config)
    source_model = ETEGRecModel(base_config)
    source_model.build_train_collator(etegrec_data)

    state = source_model._rqvae_module().state_dict()
    checked_key = next(key for key, value in state.items() if torch.is_floating_point(value))
    expected = torch.full_like(state[checked_key], 0.314159)
    state[checked_key] = expected.clone()
    checkpoint_path = tmp_path / "fixed_rqvae.pt"
    torch.save({"state_dict": state}, checkpoint_path)

    load_config = ETEGRecConfig(
        history_max_length=2,
        semantic_emb_file=str(_semantic_emb_file(tmp_path, num_items=8, dim=4)),
        rqvae_path=str(checkpoint_path),
        semantic_hidden_size=4,
        code_num=8,
        code_length=4,
        num_emb_list=(8, 8, 8),
        e_dim=4,
        layers=(4,),
        num_layers=1,
        num_decoder_layers=1,
        d_model=8,
        d_ff=16,
        num_heads=2,
        d_kv=4,
        dropout_rate=0.0,
    )
    loaded_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=load_config)
    loaded_model = ETEGRecModel(load_config)
    loaded_model.build_train_collator(loaded_data)

    loaded_value = loaded_model._rqvae_module().state_dict()[checked_key]
    torch.testing.assert_close(loaded_value, expected)


def test_etegrec_rqvae_checkpoint_load_marks_quantizers_initialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    base_config = ETEGRecConfig(
        history_max_length=2,
        semantic_emb_file=str(_semantic_emb_file(tmp_path, num_items=8, dim=4)),
        semantic_hidden_size=4,
        code_num=8,
        code_length=4,
        num_emb_list=(8, 8, 8),
        e_dim=4,
        layers=(4,),
        num_layers=1,
        num_decoder_layers=1,
        d_model=8,
        d_ff=16,
        num_heads=2,
        d_kv=4,
        dropout_rate=0.0,
    )
    etegrec_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=base_config)
    source_model = ETEGRecModel(base_config)
    source_model.build_train_collator(etegrec_data)
    checkpoint_path = tmp_path / "rqvae.pt"
    torch.save({"state_dict": source_model._rqvae_module().state_dict()}, checkpoint_path)

    original_build_rqvae = ETEGRecModel._build_rqvae

    def build_rqvae_with_uninitialized_quantizers(self):
        rqvae = original_build_rqvae(self)
        for quantizer in rqvae.rq.vq_layers:
            quantizer.initted = False
        return rqvae

    monkeypatch.setattr(ETEGRecModel, "_build_rqvae", build_rqvae_with_uninitialized_quantizers)
    load_config = ETEGRecConfig(
        history_max_length=2,
        semantic_emb_file=str(_semantic_emb_file(tmp_path, num_items=8, dim=4)),
        rqvae_path=str(checkpoint_path),
        semantic_hidden_size=4,
        code_num=8,
        code_length=4,
        num_emb_list=(8, 8, 8),
        e_dim=4,
        layers=(4,),
        num_layers=1,
        num_decoder_layers=1,
        d_model=8,
        d_ff=16,
        num_heads=2,
        d_kv=4,
        dropout_rate=0.0,
    )
    loaded_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=load_config)
    loaded_model = ETEGRecModel(load_config)
    loaded_model.build_train_collator(loaded_data)

    assert all(quantizer.initted is True for quantizer in loaded_model._rqvae_module().rq.vq_layers)


def test_etegrec_tiny_trainer_runs_train_valid_test(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = ETEGRecConfig(
        history_max_length=2,
        semantic_emb_file=str(_semantic_emb_file(tmp_path, num_items=8, dim=4)),
        semantic_hidden_size=4,
        code_num=8,
        code_length=4,
        num_emb_list=(8, 8, 8),
        e_dim=4,
        layers=(4,),
        num_layers=1,
        num_decoder_layers=1,
        d_model=8,
        d_ff=16,
        num_heads=2,
        d_kv=4,
        dropout_rate=0.0,
        num_beams=3,
        eval_topk=(3,),
    )
    etegrec_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=config)
    model = ETEGRecModel(config)
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            batch_size=2,
            shuffle=False,
            max_epochs=2,
            cycle=2,
            lr_rec=1e-3,
            lr_id=1e-3,
            monitor="recall@5",
            checkpoint=CheckpointConfig(save_best=True, save_last=True),
            eval=_full_eval_config_with_metrics(),
        )
    )

    result = trainer.run(model, etegrec_data, output_dir=tmp_path / "run")

    assert len(result["fit"]["train_history"]) == 2
    assert result["fit"]["train_history"][0]["mode"] == "rqvae"
    assert result["fit"]["train_history"][1]["mode"] == "rec"
    assert result["fit"]["checkpoint_paths"]["best"] is not None
    assert Path(result["fit"]["checkpoint_paths"]["best"]).exists()
    assert Path(result["fit"]["checkpoint_paths"]["last"]).exists()
    assert result["test"]["split"] == "test"
    assert "recall@5" in result["test"]["metrics"]


def test_etegrec_joint_eval_steps_runs_on_interval_and_last_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = _tiny_etegrec_config(tmp_path)
    etegrec_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=config)
    model = ETEGRecModel(config)
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            batch_size=2,
            shuffle=False,
            max_epochs=3,
            eval_steps=2,
            cycle=2,
            monitor="recall@5",
            checkpoint=CheckpointConfig(save_best=True, save_last=True),
            eval=_full_eval_config_with_metrics(),
        )
    )
    eval_values = iter([0.5, 0.6])
    saved_paths: list[str] = []

    monkeypatch.setattr(trainer, "create_accelerator", lambda: _DummyAccelerator())
    monkeypatch.setattr(trainer, "_train_epoch", lambda **kwargs: [float(kwargs["epoch"])])
    monkeypatch.setattr(
        trainer,
        "_run_evaluation",
        lambda *args, **kwargs: {
            "split": "valid",
            "protocol": "full",
            "loss": None,
            "metrics": {"recall@5": next(eval_values)},
            "num_batches": 1,
            "data_stats": {},
        },
    )
    monkeypatch.setattr(
        trainer,
        "_save_model_checkpoint",
        lambda model, accelerator, path: saved_paths.append(Path(path).name),
    )

    result = trainer.fit(model, etegrec_data, output_dir=tmp_path / "eval-steps")

    assert [entry["epoch"] for entry in result["train_history"]] == [1, 2, 3]
    assert [entry["epoch"] for entry in result["valid_history"]] == [2, 3]
    assert result["best_epoch"] == 3
    assert saved_paths.count("best_model.pt") == 2
    assert saved_paths.count("last_model.pt") == 3


def test_etegrec_joint_eval_steps_skipped_epochs_do_not_count_for_early_stopping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = StubDataset(StubDatasetConfig()).prepare(eval_config=_full_eval_config())
    config = _tiny_etegrec_config(tmp_path)
    etegrec_data = ETEGRecModelDataset.from_task_dataset(prepared, model_config=config)
    model = ETEGRecModel(config)
    trainer = ETEGRecTrainer(
        ETEGRecTrainerConfig(
            batch_size=2,
            shuffle=False,
            max_epochs=4,
            eval_steps=2,
            cycle=2,
            monitor="recall@5",
            early_stopping=EarlyStoppingConfig(enabled=True, patience=1, min_delta=0.0),
            checkpoint=CheckpointConfig(save_best=False, save_last=False),
            eval=_full_eval_config_with_metrics(),
        )
    )
    eval_values = iter([1.0, 0.5])

    monkeypatch.setattr(trainer, "create_accelerator", lambda: _DummyAccelerator())
    monkeypatch.setattr(trainer, "_train_epoch", lambda **kwargs: [float(kwargs["epoch"])])
    monkeypatch.setattr(
        trainer,
        "_run_evaluation",
        lambda *args, **kwargs: {
            "split": "valid",
            "protocol": "full",
            "loss": None,
            "metrics": {"recall@5": next(eval_values)},
            "num_batches": 1,
            "data_stats": {},
        },
    )

    result = trainer.fit(model, etegrec_data, output_dir=tmp_path / "eval-steps-early-stop")

    assert [entry["epoch"] for entry in result["train_history"]] == [1, 2, 3, 4]
    assert [entry["epoch"] for entry in result["valid_history"]] == [2, 4]
    assert result["best_epoch"] == 2
    assert result["stopped_early"] is True


def test_etegrec_train_collator_offsets_zero_based_item_ids_and_pads() -> None:
    records = pd.DataFrame(
        [
            {USER_ID: 0, ITEM_ID: 3, HISTORY_ITEM_IDS: (0, 1, 2)},
            {USER_ID: 1, ITEM_ID: 5, HISTORY_ITEM_IDS: (4,)},
        ]
    )

    batch = ETEGRecTrainCollator(ETEGRecConfig(history_max_length=2), prepared_data=object())(records)

    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["targets"].dtype == torch.long
    assert batch["input_ids"].tolist() == [[2, 3], [5, 0]]
    assert batch["attention_mask"].tolist() == [[True, True], [True, False]]
    assert batch["targets"].tolist() == [[4], [6]]


def test_etegrec_collator_preserves_unbounded_history_when_history_max_length_is_none() -> None:
    records = [{USER_ID: 0, ITEM_ID: 4, HISTORY_ITEM_IDS: (0, 1, 2, 3)}]

    batch = ETEGRecTrainCollator(ETEGRecConfig(history_max_length=None), prepared_data=object())(records)

    assert batch["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert batch["attention_mask"].tolist() == [[True, True, True, True]]
    assert batch["targets"].tolist() == [[5]]


def test_etegrec_eval_collator_outputs_generation_inputs_without_targets() -> None:
    records = pd.DataFrame([{USER_ID: 0, ITEM_ID: 3, HISTORY_ITEM_IDS: (0, 1)}])

    batch = ETEGRecEvalCollator(ETEGRecConfig(history_max_length=50), prepared_data=object())(records)

    assert set(batch) == {"input_ids", "attention_mask"}
    assert batch["input_ids"].tolist() == [[1, 2]]
    assert batch["attention_mask"].tolist() == [[True, True]]


def test_etegrec_collator_keeps_empty_history_shape_stable() -> None:
    records = [{USER_ID: 0, ITEM_ID: 0, HISTORY_ITEM_IDS: ()}]

    batch = ETEGRecTrainCollator(ETEGRecConfig(history_max_length=2), prepared_data=object())(records)

    assert batch["input_ids"].tolist() == [[0]]
    assert batch["attention_mask"].tolist() == [[False]]
    assert batch["targets"].tolist() == [[1]]


def test_etegrec_collator_rejects_non_positive_history_max_length() -> None:
    with pytest.raises(ValueError, match="history_max_length"):
        ETEGRecTrainCollator(ETEGRecConfig(history_max_length=0), prepared_data=object())
