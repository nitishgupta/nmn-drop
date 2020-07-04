local utils = import "utils.libsonnet";

local batch_size = utils.parse_number(std.extVar("BS"));
local input_size = utils.parse_number(std.extVar("ISIZE"));
local hidden_size = utils.parse_number(std.extVar("HSIZE"));
local num_layers = utils.parse_number(std.extVar("LAYERS"));

{
  "dataset_reader": {
      "type": "passage_attn2count_reader",
      "min_passage_length": 100,
      "max_passage_length": 600,
      "min_span_length": 5,
      "max_span_length": 15,
      "samples_per_bucket_count": 2000,
      "normalized": true,
      "withnoise": true,
  },

  "validation_dataset_reader": {
      "type": "passage_attn2count_reader",
      "min_passage_length": 100,
      "max_passage_length": 600,
      "min_span_length": 5,
      "max_span_length": 15,
      "samples_per_bucket_count": 500,
      "normalized": true,
      "withnoise": true,
  },

  "train_data_path": "",
  "validation_data_path": "",

  "model": {
      "type": "drop_pattn2count",

      "passage_attention_to_count": {
          "type": std.extVar("TYPE"),
          "input_size": input_size,
          "hidden_size": hidden_size,
          "num_layers": num_layers,
          "bidirectional": true,
          "dropout": 0.2
      },
  },

  "data_loader": {
      "batch_sampler": {
        "type": "basic",
        "sampler": {"type": "random"},
        "batch_size": batch_size,
        "drop_last": false,
      },
  },

  "trainer": {
      "checkpointer": {"num_serialized_models_to_keep": 1},
      "grad_norm": 5,
      "patience": 5,
      "cuda_device": utils.parse_number(std.extVar("GPU")),
      "num_epochs": utils.parse_number(std.extVar("EPOCHS")),
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
      "validation_metric": "+acc"
  },

  "random_seed": utils.parse_number(std.extVar("SEED")),
  "numpy_seed": utils.parse_number(std.extVar("SEED")),
  "pytorch_seed": utils.parse_number(std.extVar("SEED"))

}