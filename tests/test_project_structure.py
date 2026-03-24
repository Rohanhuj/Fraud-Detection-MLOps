from pathlib import Path


def test_expected_project_paths_exist() -> None:
    expected_paths = [
        Path("src"),
        Path("src/retraining"),
        Path("src/monitoring"),
        Path("artifacts"),
        Path("data"),
    ]

    missing = [str(path) for path in expected_paths if not path.exists()]
    assert not missing, f"Missing expected project paths: {missing}"
