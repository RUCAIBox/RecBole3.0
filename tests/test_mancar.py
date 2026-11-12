import math

import pandas as pd
import pytest
import torch

from recbole3.dataset import FrameDataset, ITEM_ID, LABEL, USER_ID, ParsedData
from recbole3.dataset.amazon2023.retrieval import Amazon2023RetrievalParser
from recbole3.model import get_model_spec
from recbole3.model.mancar import ManCARConfig, ManCARModel
from recbole3.model.mancar.data import ManCARCollator, build_swing
from recbole3.model.sequential import HISTORY_ITEM_IDS
from recbole3.run import compose_config, run_experiment
from recbole3.trainer import Trainer


def test_mancar_swing_and_collator_use_training_prefixes():
    train = pd.DataFrame({
        USER_ID: [0, 0, 0, 0, 1, 1, 1],
        ITEM_ID: [0, 1, 2, 4, 0, 1, 5],
        LABEL: [1, "1", "0", 1, 1, None, 1],
        HISTORY_ITEM_IDS: [(), (0,), (1,), (1,), (), (0,), (1,)],
    })
    config = ManCARConfig(items_per_trigger=1, history_max_length=1)
    item_ids = (2, 5, 1, 3, 6, 4)
    graph = build_swing(train, config, item_ids=item_ids)
    assert set(graph) == {2, 5}
    assert graph[2][0][0] == 5

    class Catalog:
        mancar_item_ids = item_ids

        def get_num_items(self):
            return 6

        def get_train_dataset(self):
            return FrameDataset(train)

    batch = ManCARCollator(config, Catalog(), neighbors=graph, training=True)(train)
    assert batch["history_id"].shape == (7, 50)
    assert batch["history_id"][0].count_nonzero().item() == 0
    assert batch["history_id"][1, -2:].tolist() == [0, 2]
    assert batch["item_id"].tolist() == [2, 5, 1, 6, 2, 5, 4]
    assert batch["prompt_ids"][1].tolist() == [0, 0, 0, 5]
    assert batch["neighbor_item_i2i"][1, 0].item() == 5

    model = ManCARModel(config)
    model.ensure_initialized(Catalog())
    model.core.inference = lambda feed: {
        "prediction": torch.tensor([[100, 5, 2, 4, 3, 6, 1]], dtype=torch.float)
    }
    result = model.predict(
        {k: v[:1] for k, v in batch.items()}, k=7, exclude_item_ids=torch.tensor([[1]])
    )
    assert result.tolist() == [[2, 3, 5, 0, 4, -1, -1]]


@pytest.mark.parametrize("category", ["Musical_Instruments", "Industrial_and_Scientific"])
def test_mancar_runs_with_shared_amazon_pipeline(tmp_path, monkeypatch, category):
    rows = [
        {USER_ID: user, ITEM_ID: str(item), "timestamp": ts}
        for user, items in (("a", [0, 1, 2, 3, 4]), ("b", [0, 1, 2, 5, 6]))
        for ts, item in enumerate(items)
    ]
    monkeypatch.setattr(
        Amazon2023RetrievalParser,
        "parse",
        lambda self: ParsedData(interactions=pd.DataFrame(rows)),
    )
    cfg = compose_config([
        "dataset=amazon2023_retrieval",
        f"dataset.category={category}",
        "dataset.metadata_mode=none",
        "model=mancar",
        "model.reason_step=1",
        "model.num_layers=1",
        "model.items_per_trigger=3",
        "trainer.max_epochs=1",
        "trainer.batch_size=4",
        "+trainer.dataloader_num_workers=0",
        "runtime.device=cpu",
        f"runtime.output_dir={tmp_path.as_posix()}",
    ])
    spec = get_model_spec("mancar")
    assert spec.model_cls is ManCARModel and spec.trainer_cls is Trainer
    torch.manual_seed(0)
    result = run_experiment(cfg)
    assert math.isfinite(result["fit"]["train_history"][0]["loss"])
    assert "ndcg@10" in result["test"]["metrics"]
    prepared = result["prepared_data"]
    assert result["fit"]["checkpoint_paths"]["best"] is not None
    assert prepared.mancar_item_ids == (1, 2, 3, 4, 6, 5, 7)
    frame = prepared.get_train_dataset().frame
    assert frame[HISTORY_ITEM_IDS].map(len).tolist().count(0) == 2
