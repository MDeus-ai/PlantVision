import pytest
import yaml
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

    # Tell the PlantVision system to use the temporary configs and data from temporary project root
    # By monkeypatching the PROJECT_ROOT in the actual paths.py module
    monkeypatch.setattr('src.paths.CONFIG_DIR', project_root / 'configs')
    monkeypatch.setattr('src.paths.DATA_DIR', project_root / 'data')


    # mock the `mlflow.pytorch.log_model` function to do nothing, literally😅
    with patch('mlflow.pytorch.log_model') as mock_log_model:

        # ACT & ASSERT
        try:
            # Run the entire main training function
            train_main()
        except Exception as e:
            pytest.fail(f"The training pipeline failed with an exception: {e}")

    # Confirm that model logging function was called
    mock_log_model.assert_called_once()