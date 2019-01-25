{
  "dataset_reader": {
    "type": std.extVar("DATASET_READER"),
    "lazy": true,

    "sentence_token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true,
        "namespace": "tokens"
      }
    }
  },

  "vocabulary": {
    "min_count": {
      "tokens": std.parseInt(std.extVar("TOKEN_MIN_CNT"))
    },
    "pretrained_files": {
      "tokens": std.extVar("WORD_EMBED_FILE")
    },
    "only_include_pretrained_words": true
  },

  "train_data_path": std.extVar("TRAINING_DATA_FILE")
//  "validation_data_path": std.extVar("valfile"),
//  "test_data_path": std.extVar("testfile"),
}