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
      else if tokenidx == "glovechar" then {
        "tokens": {
            "type": "single_id",
            "lowercase_tokens": true
        },
        "token_characters": {
          "type": "characters",
          "character_tokenizer": {
              "byte_encoding": "utf-8",
              "start_tokens": [259],
              "end_tokens": [260, 0, 0, 0, 0,0]
          }
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
    "type": "drop_parser",

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
      else if tokenidx == 'glovechar' then {
        "tokens": {
          "type": "embedding",
          "pretrained_file": std.extVar("BIDAF_WORDEMB_FILE"),
          "embedding_dim": 100,
          "trainable": false
        },
        "token_characters": {
          "type": "character_encoding",
          "embedding": {
            "num_embeddings": 262,
            "embedding_dim": 16
          },
          "encoder": {
            "type": "cnn",
            "embedding_dim": 16,
            "num_filters": 100,
            "ngram_filter_sizes": [
                5
            ]
          },
          "dropout": 0.2
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
    "phrase_layer": {
        "type": "lstm",
        "input_size": token_embed_dim,
        "hidden_size": 100,
        "num_layers": 1,
        "bidirectional": true,
    },
    "matrix_attention_layer": {
        "type": "linear",
        "tensor_1_dim": 128,
        "tensor_2_dim": 128,
        "combination": "x,y,x*y"
    },
    "modeling_layer": {
        "type": "lstm",
        "input_size": token_embed_dim,
        "hidden_size": 100,
        "num_layers": 1,
        "bidirectional": true,
    },

    "goldactions": utils.boolparser(std.extVar("GOLDACTIONS")),

    "bidafutils":
      if tokenidx == "bidaf" then {
        "bidaf_model_path": std.extVar("BIDAF_MODEL_TAR"),
        "bidaf_wordemb_file": std.extVar("BIDAF_WORDEMB_FILE"),
      }
    ,
    "action_embedding_dim": 100,

    "decoder_beam_search": {
      "beam_size": utils.parse_number(std.extVar("BEAMSIZE")),
    },

    "max_decoding_steps": utils.parse_number(std.extVar("MAX_DECODE_STEP")),
    "dropout": utils.parse_number(std.extVar("DROPOUT")),
//    "question_token_repr_key": std.extVar("QTK"),
//    "context_token_repr_key": std.extVar("CTK"),
    "aux_goldprog_loss": utils.boolparser(std.extVar("AUXGPLOSS")),
    "qatt_coverage_loss": utils.boolparser(std.extVar("ATTCOVLOSS")),
    "initializers":
      if utils.boolparser(std.extVar("PTREX")) == true then
      [
          ["executor_parameters.*",
             {
                 "type": "pretrained",
                 "weights_file_path": "./resources/semqa/checkpoints/hpqa/b_wsame/hpqa_parser/BS_4/OPT_adam/LR_0.001/Drop_0.2/TOKENS_glove/FUNC_snli/SIDEARG_true/GOLDAC_true/AUXGPLOSS_false/QENTLOSS_false/ATTCOV_false/best.th",
             }
          ]
      ]
      else
      [],
    "regularizer": [
      [
          ".*",
          {
              "type": "l2",
              "alpha": 1e-04
          }
      ]
    ],
  },

  "iterator": {
    "type": "basic",
    "track_epoch": true,
    "batch_size": std.extVar("BS"),
    "max_instances_in_memory": std.extVar("BS") //
  },

  "trainer": {
    "num_serialized_models_to_keep": -1,
    "grad_clipping": 10.0,
    "cuda_device": utils.parse_number(std.extVar("GPU")),
    "num_epochs": utils.parse_number(std.extVar("EPOCHS")),
    "shuffle": false,
    "optimizer": {
      "type": std.extVar("OPT"),
      "lr": utils.parse_number(std.extVar("LR"))
    },
    "summary_interval": 10,
    "validation_metric": "+f1"
  },

  "random_seed": 100,
  "numpy_seed": 100,
  "pytorch_seed": 100

}