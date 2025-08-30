import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch
from pathlib import Path

from plantvision import predict, paths


# Reuse the 'dummy_evaluation_project' fixture from conftest.py
# because it provides everything needed, a model, data (for class names), and a config.

def test_predict_handles_nonexistent_image(capsys):

    """
    Tests if the predict function gracefully handles a path to an image that does not exist.
    """

    # ARRANGE
    # Create a path that we know doesn't exist
    non_existent_path = Path("path/to/non/existent/image.jpg")


    # ACT
    # Call the function with dummy paths for the other arguments since they won't be used.
    predict.predict(
        model_checkpoint=Path("."),
        image_path=non_existent_path,
    )

    # ASSERT
    # 'capsys' is a pytest fixture that captures printed output to stdout/stderr.
    captured = capsys.readouterr()
    assert "❌ Error: No image file found at" in captured.out


def test_predict_runs_successfully(monkeypatch, dummy_evaluation_project):

    """
    A smoke test to ensure the main predict function runs end-to-end without crashing.
    """

    # ARRANGE
    # Get all the paths from the shared fixture
    project_root = dummy_evaluation_project["project_root"]
    model_path = dummy_evaluation_project["model_path"]
    data_path = dummy_evaluation_project["data_path"]
    config_dir = dummy_evaluation_project["config_dir"]

    # Create a dummy image file inside the fixture's temporary structure
    dummy_image_path = data_path / "class_a" / "predict_this.png"
    dummy_image = Image.fromarray(np.uint8(np.zeros((10, 10, 3))))
    dummy_image.save(dummy_image_path)

    # Monkeypatch to use a fake project root and config_directory
    monkeypatch.setattr(paths, 'PROJECT_ROOT', project_root)
    monkeypatch.setattr(paths, 'CONFIG_DIR', config_dir)


    # ACT & ASSERT
    # Assert that running the function does NOT raise an exception
    try:
        # Mock the print function to keep the test output clean
        with patch('builtins.print'):
            predict.predict(
                model_checkpoint=model_path,
                image_path=dummy_image_path,
            )
    except Exception as e:
        pytest.fail(f"predict.predict() raised an exception unexpectedly: {e}")