local utils = import 'utils.libsonnet';
local bert_model = "bert-base-uncased";
local batch_size = utils.parse_number(std.extVar("BS"));
local max_length = 512;
local beam_size = utils.parse_number(std.extVar("BEAMSIZE"));

{
    "dataset_reader": {
        "type": "drop_qparser_reader_bert",
        "lazy": false,
        "skip_instances": true,
        "skip_if_progtype_mismatch_anstype": false,
        "max_question_wps": 50,
        "max_transformer_length": max_length,
        "tokenizer": {
          "model_name": bert_model,
          "add_special_tokens": false,
        },
        "token_indexers": {
          "tokens": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
            "max_length": max_length
          }
        },
    },

    "validation_dataset_reader": {
        "type": "drop_qparser_reader_bert",
        "lazy": false,
        "skip_instances": false,
        "max_question_wps": 50,
        "tokenizer": {
          "model_name": bert_model,
          "add_special_tokens": false
        },
        "token_indexers": {
          "tokens": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
            "max_length": max_length
          }
        },
    },

    "train_data_path": std.extVar("TRAINING_DATA_FILE"),
    "validation_data_path": std.extVar("VAL_DATA_FILE"),

    "model": {
        "type": "ques_qparser_bert",

        "transformer_model_name": bert_model,

        "max_ques_len": 50,

        "action_embedding_dim": 100,

        "transitionfunc_attention": {
          "type": "dot_product",
          "normalize": true
        },

        "beam_size": beam_size,

        "max_decoding_steps": utils.parse_number(std.extVar("MAX_DECODE_STEP")),
        "dropout": utils.parse_number(std.extVar("DROPOUT")),

        "qattloss": utils.boolparser(std.extVar("QATTLOSS")),
        "profile_freq": utils.parse_number(std.extVar("PROFILE_FREQ")),
    },

    "data_loader": {
      "batch_sampler": {
        "type": "basic",
        "sampler": {"type": "random"},
        "batch_size": batch_size,
        "drop_last": false,
      },
    },

    "validation_data_loader": {
      "batch_sampler": {
        "type": "basic",
        "sampler": {"type": "random"},
        "batch_size": batch_size,
        "drop_last": false,
      },
    },

    "trainer": {
        "checkpointer": {"num_serialized_models_to_keep": 1},
        "grad_norm": 5,
        "patience": 10,
        "cuda_device": utils.parse_number(std.extVar("GPU")),
        "num_epochs": utils.parse_number(std.extVar("EPOCHS")),
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 2e-5
        },
        "validation_metric": "+acc",
    },

    "random_seed": utils.parse_number(std.extVar("SEED")),
    "numpy_seed": utils.parse_number(std.extVar("SEED")),
    "pytorch_seed": utils.parse_number(std.extVar("SEED"))

}
