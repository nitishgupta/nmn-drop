local parser = {
//   boolparser(x): true if x == "true" else False,
  boolparser(x):
    if x == "true" then true
    else false
};

local parse_number(x) =
  local a = std.split(x, ".");
  if std.length(a) == 1 then
    std.parseInt(a[0])
  else
    local denominator = std.pow(10, std.length(a[1]));
    local numerator = std.parseInt(a[0] + a[1]);
    local parsednumber = numerator / denominator;
    parsednumber;


{
  "dataset_reader": {
    "type": std.extVar("DATASET_READER"),
    "lazy": true,

    "token_indexers": {
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
  },

  "vocabulary": {
    "directory_path": std.extVar("VOCABDIR")
  },

  "train_data_path": std.extVar("TRAINING_DATA_FILE"),
  "validation_data_path": std.extVar("VAL_DATA_FILE"),
//  "test_data_path": std.extVar("testfile"),


  "model": {
    "type": "hotpotqa_parser",

//    "text_field_embedder": {
//      "tokens": {
//          "type": "embedding",
//          "pretrained_file": std.extVar("WORD_EMBED_FILE"),
//          "embedding_dim": 100,
//          "trainable": false
//      },
//      "token_characters": {
//          "type": "character_encoding",
//          "embedding": {
//              "num_embeddings": 262,
//              "embedding_dim": 16
//          },
//          "encoder": {
//              "type": "cnn",
//              "embedding_dim": 16,
//              "num_filters": 100,
//              "ngram_filter_sizes": [
//                  5
//              ]
//          },
//          "dropout": 0.2
//      }
//    },

    "action_embedding_dim": 100,

//    "qencoder": {
//      "type": "lstm",
//      "input_size": 200,
//      "hidden_size": 50,
//      "num_layers": 1
//    },

    // Spans of this will be used for action embedding. So final output should be equal to action_embedding_dim
    // hidden_size * dir * 2 == action_embedding_dim
    // Don't really need this now ...
//    "ques2action_encoder": {
//      "type": "lstm",
//      "input_size": 200,
//      "hidden_size": 50,
//      "num_layers": 1,
//    },
//    "quesspan_extractor": {
//      "type": "endpoint",
//      "input_dim": 50,
//    },

    "attention": {"type": "dot_product"},

    "decoder_beam_search": {
      "beam_size": parse_number(std.extVar("BEAMSIZE")),
    },

    "executor_parameters": {
//      "ques_encoder": {
//        "type": "lstm",
//        "input_size": 200,
//        "hidden_size": 50,
//        "num_layers": 1,
//        "bidirectional": true
//      },
//      "context_encoder": {
//        "type": "lstm",
//        "input_size": 200,
//        "hidden_size": 50,
//        "num_layers": 1,
//        "bidirectional": true
//      },
      "dropout": parse_number(std.extVar("DROPOUT"))
    },
    "beam_size": parse_number(std.extVar("BEAMSIZE")),
    "max_decoding_steps": parse_number(std.extVar("MAX_DECODE_STEP")),
    "dropout": parse_number(std.extVar("DROPOUT")),
    "bidaf_model_path": std.extVar("BIDAF_MODEL_TAR"),
    "bidaf_wordemb_file": std.extVar("BIDAF_WORDEMB_FILE"),
  },

  "iterator": {
    "type": "basic",
    "batch_size": std.extVar("BS"),
    "max_instances_in_memory": std.extVar("BS") //
  },

  "trainer": {
    "grad_clipping": 10.0,
    "cuda_device": parse_number(std.extVar("GPU")),
    "num_epochs": parse_number(std.extVar("EPOCHS")),
    "shuffle": false,
    "optimizer": {
      "type": std.extVar("OPT"),
      "lr": parse_number(std.extVar("LR"))
    },
    "summary_interval": 10,
    "validation_metric": "+accuracy"
  },

  "random_seed": 100,
  "numpy_seed": 100,
  "pytorch_seed": 100

}