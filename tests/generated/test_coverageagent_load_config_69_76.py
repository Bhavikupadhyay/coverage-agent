import pytest
from unittest import mock
from pathlib import Path
from coverage_agent.config import load_config, AgentConfig
import yaml

@pytest.fixture
def tmp_config_file(tmp_path):
    config_file = tmp_path / '.coverage-agent.yml'
    return config_file

def test_load_config_path(tmp_config_file):
    with open(tmp_config_file, 'w') as f:
        yaml.dump({'test_config': 'test_value'}, f)

    with mock.patch('coverage_agent.config.logger') as mock_logger:
        config = load_config(str(tmp_config_file))

    assert isinstance(config, AgentConfig)
    mock_logger.debug.assert_called_once()

def test_load_config_no_path(tmp_config_file):
    with open(tmp_config_file, 'w') as f:
        yaml.dump({'test_config': 'test_value'}, f)

    with mock.patch('coverage_agent.config.logger') as mock_logger:
        config = load_config()

    assert isinstance(config, AgentConfig)
    mock_logger.debug.assert_called_once()

def test_load_config_invalid_path(tmp_config_file):
    with mock.patch('coverage_agent.config.logger') as mock_logger:
        config = load_config(str(tmp_config_file / 'non_existent_file'))

    assert isinstance(config, AgentConfig)
    mock_logger.warning.assert_called_once()

def test_load_config_invalid_yaml(tmp_config_file):
    with open(tmp_config_file, 'w') as f:
        f.write('invalid yaml')

    with mock.patch('coverage_agent.config.logger') as mock_logger:
        config = load_config(str(tmp_config_file))

    assert isinstance(config, AgentConfig)
    mock_logger.warning.assert_called_once()

def test_load_config_parent_directory(tmp_config_file):
    parent_dir = tmp_config_file.parent
    grand_parent_dir = parent_dir.parent

    with open(tmp_config_file, 'w') as f:
        yaml.dump({'test_config': 'test_value'}, f)

    with mock.patch('pathlib.Path.cwd', return_value=grand_parent_dir):
        with mock.patch('coverage_agent.config.logger') as mock_logger:
            config = load_config()

    assert isinstance(config, AgentConfig)
    mock_logger.debug.assert_called_once()

def test_load_config_defaults(tmp_config_file):
    with mock.patch('coverage_agent.config.logger') as mock_logger:
        config = load_config()

    assert isinstance(config, AgentConfig)
    mock_logger.debug.assert_called_once()