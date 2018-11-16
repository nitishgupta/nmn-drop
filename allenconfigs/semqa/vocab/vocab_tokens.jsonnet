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
  },

  "train_data_path": std.extVar("TRAINING_DATA_FILE"),
//  "validation_data_path": std.extVar("valfile"),
//  "test_data_path": std.extVar("testfile"),
}