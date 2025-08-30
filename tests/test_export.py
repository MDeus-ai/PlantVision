import yaml
import pytest
from unittest.mock import patch

from plantvision import paths
from plantvision.export import main as export_main

def test_export_pipeline_runs_without_crashing(monkeypatch, dummy_evaluation_project):
    """
    An end-to-end smoke test for the export pipeline.
    This test verifies that the export.main function can execute successfully
    on a dummy model from our test fixture
    """

    # ARRANGE

    project_root = dummy_evaluation_project["project_root"]
    outputs_dir = dummy_evaluation_project["outputs_dir"]
    data_dir = dummy_evaluation_project["data_dir"]
    config_dir = dummy_evaluation_project["config_dir"]
    model_path = dummy_evaluation_project["model_path"]

    # Define where the output models to be saved within the temp directory
    output_dir = project_root / "models_exported"

    monkeypatch.setattr(paths, 'CONFIG_DIR', config_dir)
    monkeypatch.setattr(paths, 'OUTPUTS_DIR', outputs_dir)
    monkeypatch.setattr(paths, 'DATA_DIR', data_dir)

    # Simulate a user running the script from the command line by creating
    # a fake sys.argv list
    test_args = [
        "plantvision-export",
        "--model-checkpoint", str(model_path),
        "--output-dir", str(output_dir),
        "--quantize"  # Test the full quantization path
    ]

    # ACT & ASSERT

    # Patch sys.argv so that when export_main() is called, its internal
    # parser.parse_args() reads test_args instead of real command-line args
    with patch('sys.argv', test_args):
        try:
            # Run the entire main function from the export script.
            export_main()
        except Exception as e:
            pytest.fail(f"The export pipeline failed unexpectedly with an exception: {e}")

    with open(dummy_evaluation_project["model_config_path"], 'r') as f:
        model_variation = yaml.safe_load(f)['model']['model_variation']

    base_filename = f"plantvision_{model_variation}"

    print(f"Checking for output files in: {output_dir}")
    assert (output_dir / f"{base_filename}.fp32.onnx").is_file(), "FP32 ONNX file was not created."
    assert (output_dir / f"{base_filename}.preprocessed.onnx").is_file(), "Preprocessed ONNX file was not created."
    assert (output_dir / f"{base_filename}.int8.onnx").is_file(), "INT8 Quantized ONNX file was not created."