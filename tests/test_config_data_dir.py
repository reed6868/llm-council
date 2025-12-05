import os
import importlib


def test_data_dir_default_under_council_home(monkeypatch, tmp_path):
    # Ensure no explicit overrides
    monkeypatch.delenv("LLM_COUNCIL_HOME", raising=False)
    monkeypatch.delenv("LLM_COUNCIL_DATA_DIR", raising=False)
    # Isolate HOME so we don't touch the real home directory
    monkeypatch.setenv("HOME", str(tmp_path))

    from backend import config as config_module
    # Avoid picking up values from a real .env during this test.
    monkeypatch.setattr(config_module, "load_dotenv", lambda: None)
    importlib.reload(config_module)

    expected_home = os.path.expanduser("~/.llm-council")
    assert config_module.LLM_COUNCIL_HOME == expected_home
    assert config_module.DATA_DIR == os.path.join(expected_home, "conversations")
    assert os.path.isabs(config_module.DATA_DIR)


def test_data_dir_respects_env_override(monkeypatch, tmp_path):
    custom_dir = tmp_path / "council"
    monkeypatch.setenv("LLM_COUNCIL_DATA_DIR", str(custom_dir))

    from backend import config as config_module

    importlib.reload(config_module)

    assert config_module.DATA_DIR == str(custom_dir)
