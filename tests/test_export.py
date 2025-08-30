import pytest
import yaml
from unittest.mock import patch
from plantvision import paths

from plantvision.export import main as export_main

def test_export_pipeline_runs_without_crashing(monkeypatch, dummy_evaluation_project):
    """
    An end-to-end smoke test for the export pipeline

    This test verifies that the 'export.main' function can execute on a dummy model and produce the expected
    .onnx files without raising any exceptions.
    It covers both the FP32 export and the INT8 quantization paths
    """

    # ARRANGE
    # Get all the necessary paths from the test fixture
    project_root = dummy_evaluation_project["project_root"]
    model_path = dummy_evaluation_project["model_path"]
    output_dir = project_root / "models_exported"

    # Load the DUMMY configs from the fixture into memory
    with open(dummy_evaluation_project["model_config_path"], 'r') as f:
        dummy_model_config = yaml.safe_load(f)
    with open(dummy_evaluation_project["data_config_path"], 'r') as f:
        dummy_data_config = yaml.safe_load(f)
    with open(dummy_evaluation_project["class_names_path"], 'r') as f:
        dummy_class_names = yaml.safe_load(f)


    def mock_load_config(path):
        if 'model_config' in str(path):
            return dummy_model_config
        if 'data_config' in str(path):
            return dummy_data_config
        return None  # Fallback

    def mock_json_load(fp):
        return dummy_class_names

    # Apply the patches
    monkeypatch.setattr('plantvision.utils.load_config', mock_load_config)
    monkeypatch.setattr('json.load', mock_json_load)
    monkeypatch.setattr(paths, 'PROJECT_ROOT', project_root)

    # Mock command-line arguments that export_main's argparse will parse
    test_args = [
        "plantvision-export",
        "--model-checkpoint", str(model_path),
        "--output-dir", str(output_dir),
        "--quantize"
    ]

    # Provide arguments as if they were coming from the command line
    with patch('sys.argv', test_args):
        # ACT & ASSERT
        try:
            export_main()
        except Exception as e:
            pytest.fail(f"The export pipeline failed with an exception: {e}")

    # Assert that the expected output files were created
    base_filename = "plantvision_b0"
    assert (output_dir / f"{base_filename}.fp32.onnx").exists()
    assert (output_dir / f"{base_filename}.preprocessed.onnx").exists()
    assert (output_dir / f"{base_filename}.int8.onnx").exists()
