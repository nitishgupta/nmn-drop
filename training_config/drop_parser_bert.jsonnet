local utils = import 'utils.libsonnet';
local bert_model = "bert-base-uncased";
local batch_size = utils.parse_number(std.extVar("BS"));
local max_length = 512;

{
    "dataset_reader": {
        "type": std.extVar("DATASET_READER"),
        "lazy": false,
        "skip_instances": true,
        "skip_due_to_gold_programs": true,
        "convert_spananswer_to_num": true,
        "question_length_limit": 50,
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
        "type": std.extVar("DATASET_READER"),
        "lazy": false,
        "skip_instances": false,
        "skip_due_to_gold_programs": false,
        "question_length_limit": 50,
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
        }
    },

    "train_data_path": std.extVar("TRAINING_DATA_FILE"),
    "validation_data_path": std.extVar("VAL_DATA_FILE"),

    "model": {
        "type": "drop_parser_bert",

        "transformer_model_name": bert_model,
        "scaling_bert": utils.boolparser(std.extVar("SCALING_BERT")),

        "max_ques_len": 50,

        "transitionfunc_attention": {
          "type": "dot_product",
          "normalize": true
        },

        "passage_attention_to_span": {
            "type": "gru",
            "input_size": 4,
            "hidden_size": 20,
            "num_layers": 2,
            "bidirectional": true,
        },

        "question_attention_to_span": {
            "type": "gru",
            "input_size": 4,
            "hidden_size": 20,
            "num_layers": 3,
            "bidirectional": true,
        },

        "passage_attention_to_count": {
            "type": "gru",
            "input_size": 4,
            "hidden_size": 20,
            "num_layers": 2,
            "bidirectional": true,
        },

        "action_embedding_dim": 100,

        "beam_size": utils.parse_number(std.extVar("BEAMSIZE")),

        "max_decoding_steps": utils.parse_number(std.extVar("MAX_DECODE_STEP")),
        "dropout": utils.parse_number(std.extVar("DROPOUT")),

        "initializers": {
            "regexes": [
                ["passage_attention_to_count|passage_count_hidden2logits",
                     {
                         "type": "pretrained",
                         "weights_file_path": "./pattn2count_ckpt/best.th"
                     },
                ],
            ],
            "prevent_regexes": [".*_text_field_embedder.*"],
        },

        "auxwinloss": utils.boolparser(std.extVar("AUXLOSS")),

        "countfixed": utils.boolparser(std.extVar("COUNT_FIXED")),

        "excloss": utils.boolparser(std.extVar("EXCLOSS")),
        "qattloss": utils.boolparser(std.extVar("QATTLOSS")),
        "mmlloss": utils.boolparser(std.extVar("MMLLOSS")),
        "hardem_epoch": utils.parse_number(std.extVar("HARDEM_EPOCH")),
        "debug": utils.boolparser(std.extVar("DEBUG")),
        "profile_freq": utils.parse_number(std.extVar("PROFILE_FREQ")),
        "cuda_device": utils.parse_number(std.extVar("GPU")),
        "interpret": utils.boolparser(std.extVar("INTERPRET"))
    },

    "data_loader": {
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
        "cuda_device": utils.parse_number(std.extVar("GPU")), // [0, 1, 2, 3],
        "num_epochs": utils.parse_number(std.extVar("EPOCHS")),
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1e-5
        },
        "validation_metric": "+f1",
        // "gc_freq": utils.parse_number(std.extVar("GC_FREQ")),
    },

    "random_seed": utils.parse_number(std.extVar("SEED")),
    "numpy_seed": utils.parse_number(std.extVar("SEED")),
    "pytorch_seed": utils.parse_number(std.extVar("SEED"))

}
