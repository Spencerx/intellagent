import pytest

FAKE_LLM_ENV = {
    'openai': {'OPENAI_API_KEY': '', 'OPENAI_ORGANIZATION': '', 'OPENAI_API_BASE': ''},
    'azure': {'AZURE_OPENAI_API_KEY': '', 'AZURE_OPENAI_ENDPOINT': '', 'OPENAI_API_VERSION': ''},
    'google': {'GOOGLE_API_KEY': ''},
    'anthropic_vertex': {'PROJECT_ID': '', 'REGION': ''},
    'anthropic': {'ANTHROPIC_KEY': ''},
    'oracle': {'SERVICE_ENDPOINT': '', 'COMPARTMENT_ID': ''},
    'ollama': {'HOST': 'http://localhost:11434'},
}

@pytest.fixture(autouse=True)
def patch_llm_env(monkeypatch):
    import simulator.utils.llm_utils as lu
    monkeypatch.setattr(lu, 'LLM_ENV', FAKE_LLM_ENV)


def test_ollama_returns_chatollama():
    from simulator.utils.llm_utils import get_llm
    from langchain_ollama import ChatOllama

    config = {'type': 'ollama', 'name': 'qwen2.5:7b'}
    llm = get_llm(config)

    assert isinstance(llm, ChatOllama)


def test_ollama_uses_default_host():
    from simulator.utils.llm_utils import get_llm
    from langchain_ollama import ChatOllama

    config = {'type': 'ollama', 'name': 'qwen2.5:7b'}
    llm = get_llm(config)

    assert isinstance(llm, ChatOllama)
    assert llm.base_url == 'http://localhost:11434'


def test_ollama_uses_config_base_url():
    from simulator.utils.llm_utils import get_llm
    from langchain_ollama import ChatOllama

    config = {'type': 'ollama', 'name': 'llama3.1:8b', 'base_url': 'http://remote-machine:11434'}
    llm = get_llm(config)

    assert isinstance(llm, ChatOllama)
    assert llm.base_url == 'http://remote-machine:11434'


def test_ollama_case_insensitive():
    from simulator.utils.llm_utils import get_llm
    from langchain_ollama import ChatOllama

    config = {'type': 'Ollama', 'name': 'mistral:7b-instruct'}
    llm = get_llm(config)

    assert isinstance(llm, ChatOllama)


def test_llm_env_has_ollama_section():
    import yaml
    with open('config/llm_env.yml') as f:
        env = yaml.safe_load(f)
    assert 'ollama' in env
    assert 'HOST' in env['ollama']
