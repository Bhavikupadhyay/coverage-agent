import pytest
from unittest.mock import patch
from coverage_agent.credentials import validate_model_id

def test_validate_model_id():
    # Test case for line 88 -> line 89 (condition: not model)
    assert validate_model_id(None) is None
    assert validate_model_id("") is None

    # Test case for line 90 -> line 93
    with patch('coverage_agent.credentials._REGISTRY', [{"id": "model1"}, {"id": "model2"}]):
        assert validate_model_id("model1") is None
        assert validate_model_id("model2") is None
        assert validate_model_id("model3") is not None