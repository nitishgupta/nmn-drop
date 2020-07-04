local utils = import 'utils.libsonnet';
local bert_model = "bert-base-uncased";
local batch_size = utils.parse_number(std.extVar("BS"));
local max_length = 512;
local beam_size = utils.parse_number(std.extVar("BEAMSIZE"));
local supervised_epochs = utils.parse_number(std.extVar("SUPEPOCHS"));
local bio_tagging = utils.boolparser(std.extVar("BIO_TAG"));
local bio_label_scheme = std.extVar("BIO_LABEL");
local qp_encoding_style = std.extVar("QP_ENC");
local qrepr_style = std.extVar("Q_REPR");

{
    "dataset_reader": {
        "type": "drop_reader_bert_v2",
        "lazy": false,
        "skip_instances": true,
        "skip_if_progtype_mismatch_anstype": false,
        "convert_spananswer_to_num": true,
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
        "bio_tagging": bio_tagging,
        "bio_label_scheme": bio_label_scheme,
    },

    "validation_dataset_reader": {
        "type": "drop_reader_bert_v2",
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
        "bio_tagging": bio_tagging,
        "bio_label_scheme": bio_label_scheme,
    },

    "train_data_path": std.extVar("TRAINING_DATA_FILE"),
    "validation_data_path": std.extVar("VAL_DATA_FILE"),

    "model": {
        "type": "drop_parser_bert",

        "bio_tagging": bio_tagging,
        "bio_label_scheme": bio_label_scheme,

        "transformer_model_name": bert_model,
        "qp_encoding_style": qp_encoding_style,
        "qrepr_style": qrepr_style,

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

        "passage_attention_to_count": {
            "type": "gru",
            "input_size": 4,
            "hidden_size": 20,
            "num_layers": 2,
            "bidirectional": true,
        },

        "action_embedding_dim": 100,

        "beam_size": beam_size,

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
//                ["passage_attention_to_span|passage_bio_predictor",
//                     {
//                         "type": "pretrained",
//                         "weights_file_path": "./pattn2bio_BIO_v1_4000/best.th"
//                     },
//                ],
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

//    "data_loader": {
//      "batch_sampler": {
//        "type": "basic",
//        "sampler": {"type": "random"},
//        "batch_size": batch_size,
//        "drop_last": false,
//      },
//    },

    "data_loader": {
      "sampler": {
        "type": "curriculum",
        "supervised_epochs": supervised_epochs,
      },
      "batch_size": batch_size,
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
        "cuda_device": utils.parse_number(std.extVar("GPU")), // [0, 1, 2, 3],
        "num_epochs": utils.parse_number(std.extVar("EPOCHS")),
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 2e-5
        },
        "validation_metric": "+f1",
    },

    "random_seed": utils.parse_number(std.extVar("SEED")),
    "numpy_seed": utils.parse_number(std.extVar("SEED")),
    "pytorch_seed": utils.parse_number(std.extVar("SEED"))

}
