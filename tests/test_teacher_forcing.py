import pytest
import torch
from rxn_negative_learning.models.rl_transformer.reinforce_lightning import (
    ReinforceLitVanillaTransformer,
)
from rxn_negative_learning.models.scorers.ideal_scorer import IdealScorer
from rxn_negative_learning.models.tokenization import SmilesTokenizer

from tests.conftest import set_randomness


@pytest.fixture(scope="session")
def vocab_file(tmp_path_factory):
    vocabulary = [
        "[PAD]",
        "[unused1]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "c",
        "C",
        "(",
        ")",
        "O",
        "Na",
        "Cl",
        "Mg",
    ]
    tmp_vocab_file = tmp_path_factory.mktemp("data") / "vocab.txt"
    with open(tmp_vocab_file, "w") as f:
        f.write("\n".join(vocabulary))
    return tmp_vocab_file


def test_model_training_step_no_teacher_forcing(
    vocab_file,
):
    """Tests the ability to make training step"""
    set_randomness()
    model_args = {
        "model_name_or_path": None,
        "model_config_name": None,
        "vocabulary": vocab_file,
        "embedding_dim": 8,
        "ffnn_hidden_dim": 3,
        "dropout": 0.0,
        "activation": "relu",
        "num_attention_heads": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "init_std": 0.02,
        "max_position_embeddings": 5000,
        "num_beams": 1,
        "max_length": 10,
        "no_teacher_forcing": True,
        "learning_rate": 1e-4,
        "regularization": None,
        "with_baseline": None,
        "baseline_targets_file": None,
        "shift": 0.0,
        "regularization_beta": 1,
    }

    pos_samples = ["CCCCCCCCCCC>>"]
    neg_samples = []
    tokenizer = SmilesTokenizer(model_args["vocabulary"])
    rllit_model = ReinforceLitVanillaTransformer(
        model_args=model_args,
        tokenizer=tokenizer,
        scorer=IdealScorer(pos_samples, neg_samples),
    )

    # creating dummy input and output of size (batch_size, sequence_length)
    sequence_length = 11
    batch_size = 2
    input_ids = torch.ones((batch_size, sequence_length)).long() * 7
    output_ids = torch.ones((batch_size, sequence_length)).long() * 6

    # Pad the decoder token ids
    for i in range(1, 4):
        output_ids[0][-i] = rllit_model.model.config.pad_token_id
        output_ids[1][-i] = rllit_model.model.config.pad_token_id

    decoder_padding_mask = output_ids.eq(rllit_model.model.config.pad_token_id)
    # Define first token of the decoder id to be the start of sequence one
    output_ids[0][0] = rllit_model.model.config.bos_token_id
    output_ids[1][0] = rllit_model.model.config.bos_token_id
    scores = torch.ones((batch_size,)).long()

    # make a forward pass
    output = rllit_model.training_step(
        {
            "encoder_input_ids": input_ids,
            "decoder_input_ids": output_ids,
            "decoder_padding_mask": decoder_padding_mask,
            "score": scores,
            "idx": torch.ones((batch_size,)).long(),
        },
        batch_idx=0,
    )

    before_outputs = rllit_model.model.decoder_outputs
    assert output.detach() == pytest.approx(51.14504623413086)

    # Check that using the predicted ids as labels with teacher forcing gives the same result
    rllit_model.model_args["no_teacher_forcing"] = True
    output_ids = torch.tensor([
        [3, 9, 9, 9, 8, 8, 8, 9, 9, 9, 9],
        [3, 9, 9, 9, 8, 8, 8, 9, 9, 9, 9],
    ])
    decoder_padding_mask = output_ids.eq(rllit_model.model.config.pad_token_id)

    output = rllit_model.training_step(
        {
            "encoder_input_ids": input_ids,
            "decoder_input_ids": output_ids,
            "decoder_padding_mask": decoder_padding_mask,
            "score": scores,
            "idx": torch.ones((batch_size,)).long(),
        },
        batch_idx=0,
    )
    assert torch.allclose(rllit_model.model.decoder_outputs, before_outputs)
    # The loss is also the same becuase I don't have the baseline model that has changed
    assert output.detach() == pytest.approx(51.14504623413086)
