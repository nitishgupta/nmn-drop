local parser = {
//   boolparser(x): true if x == "true" else False,
  boolparser(x):
    if x == "true" then true
    else false
};

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
    "extend": parser.boolparser(std.extVar("EXTEND_VOCAB")),
    "directory_path": std.extVar("EXISTING_VOCAB_DIR")
  },

  "train_data_path": std.extVar("TRAINING_DATA_FILE")
}