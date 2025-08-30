from unittest.mock import patch, MagicMock
from plantvision import evaluate, paths


def test_evaluate_script_runs_and_creates_outputs(monkeypatch, dummy_evaluation_project):
    """
    Integration test to ensure the evaluate script runs and produces output files.
    This test mocks the actual plotting to avoid GUI issues in CI.
    """

    # ARRANGE
    # Unpack the dictionary returned by the fixture for clarity
    project_root = dummy_evaluation_project["project_root"]
    model_path = dummy_evaluation_project["model_path"]
    data_path = dummy_evaluation_project["data_path"]
    config_dir = dummy_evaluation_project["config_dir"]

    # Use monkeypatch to temporarily change the project's directories
    # This ensures outputs are written to the temporary dir, not the real project dir.
    monkeypatch.setattr(paths, 'PROJECT_ROOT', project_root)
    monkeypatch.setattr(paths, 'CONFIG_DIR', config_dir)

    # Mock seaborn and matplotlib to prevent them from opening a display window,
    # which would fail in a headless CI environment.
    mock_heatmap = MagicMock()
    mock_savefig = MagicMock()

    with patch('seaborn.heatmap', mock_heatmap), \
            patch('matplotlib.pyplot.savefig', mock_savefig):
        # ACT
        evaluate.evaluate(
            model_checkpoint=model_path,
            data_path=data_path,
        )

    # ASSERT
    # Check that the output files were actually created in the temporary directory
    output_dir = project_root / "outputs"
    assert (output_dir / "classification_report.txt").exists()

    # Assert that our mocked plotting functions were called with the correct arguments.
    # This proves the code path was executed even though it didn't plot.
    mock_heatmap.assert_called_once()
    mock_savefig.assert_called_once_with(output_dir / "confusion_matrix.png")