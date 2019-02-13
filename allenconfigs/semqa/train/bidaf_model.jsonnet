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

  "model": {
    "type": "bidaf_model",

    "bool_bilinear": {
      "type": "bilinear",
      "tensor_1_dim": 200,
      "tensor_2_dim": 200,
    },
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