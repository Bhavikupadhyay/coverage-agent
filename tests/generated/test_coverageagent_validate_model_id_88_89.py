import pytest
from unittest.mock import patch
from coverage_agent.credentials import validate_model_id

@pytest.mark.parametrize("model", ["", None])
def test_validate_model_id_empty_model(model):
    assert validate_model_id(model) is None

@pytest.mark.parametrize("model", ["valid_model", "another_model"])
@patch("coverage_agent.credentials._REGISTRY", return_value=[])
def test_validate_model_id_invalid_model(mock_registry, model):
    assert validate_model_id(model) is not None