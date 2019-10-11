local utils = import 'utils.libsonnet';

{
    "dataset_reader": {
        "type": std.extVar("DATASET_READER"),
        "lazy": false,
        "skip_instances": true,
        "skip_due_to_gold_programs": true,
        "convert_spananswer_to_num": true,
        "question_length_limit": 50,
        "pretrained_model": "bert-base-uncased",
        "token_indexers": {
          "tokens": {
            "type": "bert-drop",
            "pretrained_model": "bert-base-uncased"
          }
        }
    },

    "validation_dataset_reader": {
        "type": std.extVar("DATASET_READER"),
        "lazy": false,
        "skip_instances": false,
        "skip_due_to_gold_programs": false,
        "question_length_limit": 50,
        "pretrained_model": "bert-base-uncased",
        "token_indexers": {
          "tokens": {
            "type": "bert-drop",
            "pretrained_model": "bert-base-uncased"
          }
        }
    },

    "train_data_path": std.extVar("TRAINING_DATA_FILE"),
    "validation_data_path": std.extVar("VAL_DATA_FILE"),

    "model": {
        "type": "drop_parser_bert",

        "pretrained_bert_model": "bert-base-uncased",

        "max_ques_len": 50,

        "transitionfunc_attention": {
          "type": "dot_product",
          "normalize": true
        },

        "passage_attention_to_span": {
            "type": "gru",
            "input_size": 4,
            "hidden_size": 20,
            "num_layers": 3,
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

        "initializers":
        [
            ["passage_attention_to_count|passage_count_hidden2logits",
                 {
                     "type": "pretrained",
                     "weights_file_path": "./pattn2count_ckpt/best.th"
                 },
            ],
            [".*_text_field_embedder.*", "prevent"]
        ],

        "auxwinloss": utils.boolparser(std.extVar("AUXLOSS")),

        "countfixed": utils.boolparser(std.extVar("COUNT_FIXED")),

        "denotationloss": utils.boolparser(std.extVar("DENLOSS")),
        "excloss": utils.boolparser(std.extVar("EXCLOSS")),
        "qattloss": utils.boolparser(std.extVar("QATTLOSS")),
        "mmlloss": utils.boolparser(std.extVar("MMLLOSS")),
        "debug": utils.boolparser(std.extVar("DEBUG"))
    },

    "iterator": {
        "type": "filter",
        "track_epoch": true,
        "batch_size": std.extVar("BS"),
//      "max_instances_in_memory":
        "filter_instances": utils.boolparser(std.extVar("SUPFIRST")),
        "filter_for_epochs": utils.parse_number(std.extVar("SUPEPOCHS")),
    },

    "validation_iterator": {
        "type": "basic",
        "track_epoch": true,
        "batch_size": std.extVar("BS")
    },


    "trainer": {
        "num_serialized_models_to_keep": 2,
        "grad_norm": 5,
        "patience": 20,
        "cuda_device":  utils.parse_number(std.extVar("GPU")), // [0, 1, 2, 3],
        "num_epochs": utils.parse_number(std.extVar("EPOCHS")),
        "shuffle": true,
        "optimizer": {
            "type": "bert_adam",
            "lr": 1e-5
        },
        "summary_interval": 100,
        "should_log_parameter_statistics": false,
        "validation_metric": "+f1"
    },

    "random_seed": utils.parse_number(std.extVar("SEED")),
    "numpy_seed": utils.parse_number(std.extVar("SEED")),
    "pytorch_seed": utils.parse_number(std.extVar("SEED"))

}
