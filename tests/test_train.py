import pytest
from unittest.mock  import patch
from PlantVision.train import main as train_main

def test_training_pipeline_runs_without_crashing(monkeypatch, dummy_evaluation_project):

    """
    An end-to-end smoke test for the training pipeline.

    This test verifies that the `train.main` function can execute a  very short training run on a dummy
    dataset without raising any exceptions. It checks the integration of data loading, model instantiation,
    the training loop, and MLflow logging

    :param monkeypatch:
    :param dummy_evaluation_project:
    """

    # ARRANGE
    # Get the paths to the temporary project from the dummy_evaluation_project fixture in conftest.py
    project_root = dummy_evaluation_project["project_root"]

   # Change directories to fake directories
    monkeypatch.setattr('PlantVision.paths.PROJECT_ROOT', project_root)
    monkeypatch.setattr('PlantVision.paths.CONFIG_DIR', project_root / 'configs')
    monkeypatch.setattr('PlantVision.paths.DATA_DIR', project_root / 'data')



    # mock the `mlflow.log_artifacts` function to do nothing, literally😅
    with patch('mlflow.log_artifacts') as mock_log_artifacts:

        # ACT & ASSERT
        try:
            # Run the entire main training function
            train_main()
        except Exception as e:
            pytest.fail(f"The training pipeline failed with an exception: {e}")

    # Confirm that model logging function was called
    mock_log_artifacts.assert_called_once()