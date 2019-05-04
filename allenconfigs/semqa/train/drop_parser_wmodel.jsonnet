local utils = import 'utils.libsonnet';

// This can be either 1) glove 2) bidaf 3) elmo
local tokenidx = std.extVar("TOKENIDX");


local token_embed_dim =
  if tokenidx == "glove" then 100
  else if tokenidx == "bidaf" then 200
  else if tokenidx == "elmo" then 1024
  else if tokenidx == "glovechar" then 200
  else if tokenidx == "elmoglove" then 1124;


local attendff_inputdim =
  if tokenidx == "glove" then 100
  else if tokenidx == "bidaf" then 200
  else if tokenidx == "elmo" then 1024
  else if tokenidx == "glovechar" then 200
  else if tokenidx == "elmoglove" then 1124;


local compareff_inputdim =
  if tokenidx == "glove" then 200
  else if tokenidx == "bidaf" then 400
  else if tokenidx == "elmo" then 2048
  else if tokenidx == "glovechar" then 400
  else if tokenidx == "elmoglove" then 2248;


{
    "dataset_reader": {
        "type": std.extVar("DATASET_READER"),
        "lazy": false,
        "skip_instances": true,
        "skip_due_to_gold_programs": true,
        "convert_spananswer_to_num": true,
        "passage_length_limit": 400,
        "question_length_limit": 100,
        "token_indexers":
            if tokenidx == "elmo" then {
                "elmo": {
                  "type": "elmo_characters"
                }
            }
            else if tokenidx == "qanet" then {
                "tokens": {
                    "type": "single_id",
                    "lowercase_tokens": true
                },
                "token_characters": {
                    "type": "characters",
                    "min_padding_length": 5
                }
            }
            else if tokenidx == "elmoglove" then {
                "elmo": {
                  "type": "elmo_characters"
                },
                "tokens": {
                  "type": "single_id",
                  "lowercase_tokens": true
                }
            }
        ,
    },

    "validation_dataset_reader": {
        "type": std.extVar("DATASET_READER"),
        "lazy": false,
        "skip_instances": false,
        "token_indexers":
            if tokenidx == "elmo" then {
                "elmo": {
                  "type": "elmo_characters"
                }
            }
            else if tokenidx == "qanet" then {
                "tokens": {
                    "type": "single_id",
                    "lowercase_tokens": true
                },
                "token_characters": {
                    "type": "characters",
                    "min_padding_length": 5
                }
            }
            else if tokenidx == "elmoglove" then {
                "elmo": {
                  "type": "elmo_characters"
                },
                "tokens": {
                  "type": "single_id",
                  "lowercase_tokens": true
                }
            }
        ,
    },

    "vocabulary": {
        "pretrained_files": {
            "tokens": std.extVar("WORDEMB_FILE"),
        }
    },

    "train_data_path": std.extVar("TRAINING_DATA_FILE"),
    "validation_data_path": std.extVar("VAL_DATA_FILE"),
  //  "test_data_path": std.extVar("testfile"),


    "model": {
         "type": "drop_parser_wmodel",

        "text_field_embedder":
          if tokenidx == "elmo" then {
            "elmo": {
              "type": "elmo_token_embedder",
              "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
              "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
              "do_layer_norm": false,
              "dropout": 0.5
            }
          }
          else if tokenidx == 'qanet' then {
            "tokens": {
              "type": "embedding",
              "pretrained_file": std.extVar("WORDEMB_FILE"),
              "embedding_dim": utils.parse_number(std.extVar("WEMB_DIM")),
              "trainable": false
            },
            "token_characters": {
                "type": "character_encoding",
                "embedding": {
                    "embedding_dim": 64
                },
                "encoder": {
                    "type": "cnn",
                    "embedding_dim": 64,
                    "num_filters": 200,
                    "ngram_filter_sizes": [
                        5
                    ]
                }
            }
          }
          else if tokenidx == "elmoglove" then {
            "elmo": {
              "type": "elmo_token_embedder",
              "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
              "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
              "do_layer_norm": false,
              "dropout": 0.5
            },
            "tokens": {
              "type": "embedding",
              "pretrained_file": std.extVar("BIDAF_WORDEMB_FILE"),
              "embedding_dim": 100,
              "trainable": false
            },
          }
        ,

        "transitionfunc_attention": {
          "type": "dot_product",
          "normalize": true
        },

        "num_highway_layers": 2,

//        "phrase_layer": {
//            "type": "qanet_encoder",
//            "input_dim": 128,
//            "hidden_dim": 128,
//            "attention_projection_dim": 128,
//            "feedforward_hidden_dim": 128,
//            "num_blocks": 1,
//            "num_convs_per_block": 4,
//            "conv_kernel_size": 7,
//            "num_attention_heads": 8,
//            "dropout_prob": 0.1,
//            "layer_dropout_undecayed_prob": 0.1,
//            "attention_dropout_prob": 0
//        },

        "phrase_layer": {
            "type": "gru",
            "input_size": 128,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": true
        },

        "matrix_attention_layer": {
            "type": "linear",
            "tensor_1_dim": 128,
            "tensor_2_dim": 128,
            "combination": "x,y,x*y"
        },

        "modeling_layer": {
            "type": "gru",
            "input_size": 128,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": true
        },


//        "modeling_layer": {
//            "type": "qanet_encoder",
//            "input_dim": 128,
//            "hidden_dim": 128,
//            "attention_projection_dim": 128,
//            "feedforward_hidden_dim": 128,
//            "num_blocks": 7,
//            "num_convs_per_block": 2,
//            "conv_kernel_size": 5,
//            "num_attention_heads": 8,
//            "dropout_prob": 0.1,
//            "layer_dropout_undecayed_prob": 0.1,
//            "attention_dropout_prob": 0
//        },

    //    "passage_token_to_date": {
    //        "type": "stacked_self_attention",
    //        "input_dim": 128,
    //        "hidden_dim": 128,
    //        "projection_dim": 128,
    //        "feedforward_hidden_dim": 256,
    //        "num_layers": 3,
    //        "num_attention_heads": 4,
    //    },

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
            "num_layers": 3,
            "bidirectional": true,
        },

        "action_embedding_dim": 100,

        "beam_size": utils.parse_number(std.extVar("BEAMSIZE")),

        "qp_sim_key": std.extVar("QP_SIM_KEY"),
        "sim_key": std.extVar("SIM_KEY"),

        "max_decoding_steps": utils.parse_number(std.extVar("MAX_DECODE_STEP")),
        "dropout": utils.parse_number(std.extVar("DROPOUT")),

        "regularizer": [
          [
              ".*",
              {
                  "type": "l2",
                  "alpha": 1e-07,
              }
          ]
        ],

//        "initializers":
//        [
//            ["passage_attention_to_count|passage_count_predictor",
//                 {
//                     "type": "pretrained",
//                     "weights_file_path": "./resources/semqa/checkpoints/savedmodels/count_pretrn_nobias/best.th"
//                 },
//            ],
//            [".*_text_field_embedder.*", "prevent"]
//        ],

        "goldactions": utils.boolparser(std.extVar("GOLDACTIONS")),
        "goldprogs": utils.boolparser(std.extVar("GOLDPROGS")),
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
        "num_serialized_models_to_keep": 10,
        "grad_norm": 5,
        "patience": 20,
        "cuda_device": utils.parse_number(std.extVar("GPU")),
        "num_epochs": utils.parse_number(std.extVar("EPOCHS")),
        "shuffle": true,
        "optimizer": {
            "type": "adam",
            "lr": utils.parse_number(std.extVar("LR")),
            "betas": [
                0.8,
                0.999
            ],
            "eps": 1e-07
          },
        "moving_average": {
            "type": "exponential",
            "decay": 0.9999
        },
        "summary_interval": 100,
        "should_log_parameter_statistics": false,
        "validation_metric": "+f1"
    },

    "random_seed": utils.parse_number(std.extVar("SEED")),
    "numpy_seed": utils.parse_number(std.extVar("SEED")),
    "pytorch_seed": utils.parse_number(std.extVar("SEED"))

}