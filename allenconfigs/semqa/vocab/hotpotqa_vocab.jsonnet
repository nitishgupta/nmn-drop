local parser = {
//   boolparser(x): true if x == "true" else False,
  boolparser(x):
    if x == "true" then true
    else false
};

local tokenidx = std.extVar("TOKENIDX");

{
  "dataset_reader": {
    "type": std.extVar("DATASET_READER"),
    "lazy": true,
    "wsideargs": parser.boolparser(std.extVar("W_SIDEARGS")),

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
      else if tokenidx == "glovechar" then
      {
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
    ,
  },

  "vocabulary":
    if tokenidx == "glove" then {
      "pretrained_files": {
        "tokens": std.extVar("WORD_EMBED_FILE")
      },
      "only_include_pretrained_words": false,
    }
    else if tokenidx == "bidaf" then {
      "extend": parser.boolparser(std.extVar("EXTEND_VOCAB")),
      "directory_path": std.extVar("EXISTING_VOCAB_DIR")
    }
    else if tokenidx == "elmo" then {
    }
    else if tokenidx == "glovechar" then {
      "pretrained_files": {
        "tokens": std.extVar("WORD_EMBED_FILE")
      },
      "only_include_pretrained_words": false,
    }
  ,

  "train_data_path": std.extVar("TRAINING_DATA_FILE")
}