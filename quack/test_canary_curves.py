"""Canary doses repeat every chunk of selected training stays, and nothing else."""

import pytest

from scripts.run_canary_curves import expand_records, choose_canaries


def records():
    return [
        {"centre": "miiv", "stay_id": 1, "split": "train", "chunk": 0},
        {"centre": "miiv", "stay_id": 1, "split": "train", "chunk": 1},
        {"centre": "miiv", "stay_id": 2, "split": "train", "chunk": 0},
        {"centre": "miiv", "stay_id": 3, "split": "val", "chunk": 0},
        {"centre": "miiv", "stay_id": 4, "split": "test", "chunk": 0},
    ]


def test_expansion_repeats_complete_stays_only():
    source = records()
    selected = choose_canaries(source[:2], count=1, seed=0)
    expanded = expand_records(source, selected, factor=5)
    assert len(expanded) == 13
    assert sum(row["stay_id"] == 1 for row in expanded) == 10
    assert [row for row in expanded if row["split"] != "train"] == source[-2:]
    assert len(source) == 5


def test_factor_one_keeps_original_record_order():
    source = records()
    selected = choose_canaries(source[:2], count=1, seed=0)
    assert expand_records(source, selected, factor=1) == source


@pytest.mark.parametrize("factor", [0, -1, 2.5])
def test_expansion_rejects_invalid_doses(factor):
    with pytest.raises(ValueError):
        expand_records(records(), {"not-a-training-stay"}, factor)


def test_selection_rejects_insufficient_and_heldout_keys():
    with pytest.raises(ValueError):
        choose_canaries(records(), count=3, seed=0)
    with pytest.raises(ValueError):
        expand_records(records(), {"not-a-training-stay"}, factor=5)


def test_selection_is_reproducible_and_order_independent():
    source = records()
    assert choose_canaries(source, 2, 0) == choose_canaries(
        list(reversed(source)), 2, 0
    )


def _share_tensor(queue, complete):
    import torch

    queue.put(torch.arange(4))
    complete.wait(timeout=10)


def test_short_runtime_directory_supports_tensor_file_descriptor_sharing(
    tmp_path, monkeypatch
):
    import multiprocessing
    import os
    import tempfile
    import torch
    from scripts.run_canary_curves import configure_temporary_directory

    inherited = tmp_path / ("long" * 40)
    inherited.mkdir()
    monkeypatch.setenv("TMPDIR", str(inherited))
    monkeypatch.setattr(tempfile, "tempdir", str(inherited))
    directory = configure_temporary_directory()
    assert tempfile.gettempdir() == str(directory)
    assert (
        len(os.fsencode(str(directory / "pymp-12345678" / "listener-12345678"))) <= 107
    )
    context = multiprocessing.get_context("fork")
    queue, complete = context.Queue(), context.Event()
    process = context.Process(target=_share_tensor, args=(queue, complete))
    process.start()
    try:
        result = queue.get(timeout=10).clone()
        assert torch.equal(result, torch.arange(4))
    finally:
        complete.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        queue.close()
    assert process.exitcode == 0


def test_discrete_training_passes_upstream_dataset_override_argument(
    tmp_path, monkeypatch
):
    """The updated upstream trainer requires data_path even when not overriding it."""
    from types import SimpleNamespace

    from scripts import run_canary_curves

    captured = []
    monkeypatch.setattr(
        run_canary_curves, "saved_config", lambda root: root / "reference.yaml"
    )
    monkeypatch.setattr(
        run_canary_curves,
        "historical_testbed",
        lambda: SimpleNamespace(train=lambda args: captured.append(args.data_path)),
    )
    cfg = SimpleNamespace(representation="discrete", factor=5, count=500)
    run_canary_curves.train(cfg, tmp_path)
    assert captured == [None]
