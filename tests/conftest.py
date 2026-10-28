import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="run tests that export real pretrained models")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def tiny_distilbert(n_layers=4, seed=0):
    import torch
    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    torch.manual_seed(seed)
    config = DistilBertConfig(vocab_size=200, dim=32, n_layers=n_layers, n_heads=2, hidden_dim=64,
                              max_position_embeddings=64, num_labels=2)
    return DistilBertForSequenceClassification(config).eval()


def tiny_bert(num_hidden_layers=4, seed=0):
    import torch
    from transformers import BertConfig, BertForSequenceClassification

    torch.manual_seed(seed)
    config = BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=num_hidden_layers,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
                        num_labels=3)
    return BertForSequenceClassification(config).eval()


@pytest.fixture(scope="session")
def tiny_export(tmp_path_factory):
    """One verified tiny DistilBERT export shared by read-only tests."""
    from crowdio.export.exporter import export_model

    out = tmp_path_factory.mktemp("exports")
    return export_model(tiny_distilbert(), "tiny-distil", out, log=lambda *_: None)
