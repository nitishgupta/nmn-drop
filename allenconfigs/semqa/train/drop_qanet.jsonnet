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
    "lazy": true,

    "token_indexers":
      if tokenidx == "glove" then {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true
        }
      }
      else if tokenidx == "bidaf" then {
        "tokens": {
            "type": "single_id",
            "lowercase_tokens": true
        },
        "token_characters": {
          "type": "characters",
          "character_tokenizer": {
              "byte_encoding": "utf-8",
              "start_tokens": [
                  259
              ],
              "end_tokens": [
                  260,
                  0,
                  0,
                  0,
                  0,
                  0
              ]
          }
        }
      }
      else if tokenidx == "elmo" then {
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

//  "vocabulary": {
//    "directory_path": std.extVar("VOCABDIR")
//  },

  "train_data_path": std.extVar("TRAINING_DATA_FILE"),
  "validation_data_path": std.extVar("VAL_DATA_FILE"),
//  "test_data_path": std.extVar("testfile"),


  "model": {
    "type": "drop_qanet",

    "text_field_embedder":
      if tokenidx == "glove" then {
        "tokens": {
          "type": "embedding",
          "pretrained_file": std.extVar("BIDAF_WORDEMB_FILE"),
          "embedding_dim": 100,
          "trainable": false
        },
      }
      else if tokenidx == "bidaf" then {
        "token_embedders": std.manifestPython(null)
      }
      else if tokenidx == "elmo" then {
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
          "embedding_dim": 300,
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

    "num_highway_layers": 2,

    "phrase_layer": {
        "type": "qanet_encoder",
        "input_dim": 128,
        "hidden_dim": 128,
        "attention_projection_dim": 128,
        "feedforward_hidden_dim": 128,
        "num_blocks": 1,
        "num_convs_per_block": 4,
        "conv_kernel_size": 7,
        "num_attention_heads": 8,
        "dropout_prob": 0.1,
        "layer_dropout_undecayed_prob": 0.1,
        "attention_dropout_prob": 0
    },
    "matrix_attention_layer": {
        "type": "linear",
        "tensor_1_dim": 128,
        "tensor_2_dim": 128,
        "combination": "x,y,x*y"
    },
    "modeling_layer": {
        "type": "qanet_encoder",
        "input_dim": 128,
        "hidden_dim": 128,
        "attention_projection_dim": 128,
        "feedforward_hidden_dim": 128,
        "num_blocks": 7,
        "num_convs_per_block": 2,
        "conv_kernel_size": 5,
        "num_attention_heads": 8,
        "dropout_prob": 0.1,
        "layer_dropout_undecayed_prob": 0.1,
        "attention_dropout_prob": 0
    },

    "dropout": utils.parse_number(std.extVar("DROPOUT")),

    "regularizer": [
      [
          ".*",
          {
              "type": "l2",
              "alpha": 1e-07
          }
      ]
    ],

    "debug": utils.boolparser(std.extVar("DEBUG"))
  },

  "iterator": {
      "type": "basic",
      "track_epoch": true,
      "batch_size": std.extVar("BS"),
      "max_instances_in_memory": std.extVar("BS") //
  },

  "trainer": {
      "num_serialized_models_to_keep": -1,
      "grad_norm": 5,
      "patience": 10,
      "cuda_device": utils.parse_number(std.extVar("GPU")),
      "num_epochs": utils.parse_number(std.extVar("EPOCHS")),
      "shuffle": false,
      "optimizer": {
          "type": "adam",
          "lr": 0.001,
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
      "summary_interval": 10,
      "validation_metric": "+f1"
  },

  "random_seed": 100,
  "numpy_seed": 100,
  "pytorch_seed": 100

}