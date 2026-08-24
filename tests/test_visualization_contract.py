from micro_dl import EpochResult, TrainResult, plot_training


def test_training_result_produces_headless_artifact_at_explicit_path(tmp_path):
    result = TrainResult(
        (EpochResult(1.0, 1.2), EpochResult(0.8, 0.9)),
        best_epoch=1,
    )
    destination = tmp_path / "reports" / "training.png"
    second = tmp_path / "reports" / "training-again.png"

    returned = plot_training(result, destination)
    plot_training(result, second)

    assert returned == destination
    assert destination.exists()
    assert destination.read_bytes().startswith(b"\x89PNG")
    assert destination.stat().st_size > 100
    assert destination.read_bytes() == second.read_bytes()
