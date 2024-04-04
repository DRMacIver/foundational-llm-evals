import pytest


def pytest_generate_tests(metafunc):
    if "model" in metafunc.fixturenames:
        models = ["llama2", "mistral"]
        config = metafunc.config

        if config.getoption("--run-tests-on-claude"):
            models.extend(
                [
                    "claude-3-haiku-20240307",
                    "claude-3-sonnet-20240229",
                    "claude-3-opus-20240229",
                ]
            )
        if config.getoption("--run-tests-on-gpt"):
            models.extend(["gpt-3.5-turbo", "gpt-4-turbo-preview"])
        metafunc.parametrize("model", models)


def pytest_addoption(parser):
    parser.addoption(
        "--run-tests-on-claude",
        action="store_true",
        default=False,
        help="Run tests on Claude",
    )

    parser.addoption(
        "--run-tests-on-gpt",
        action="store_true",
        default=False,
        help="Run tests on Claude",
    )
