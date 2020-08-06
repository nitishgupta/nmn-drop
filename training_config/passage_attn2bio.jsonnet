local utils = import 'utils.libsonnet';

local batch_size = utils.parse_number(std.extVar("BS"));
local input_size = utils.parse_number(std.extVar("ISIZE"));
local hidden_size = utils.parse_number(std.extVar("HSIZE"));
local num_layers = utils.parse_number(std.extVar("LAYERS"));

{
  "dataset_reader": {
      "type": "passage_attn2bio_reader",
      "joint_count": false,
      "count_samples_per_bucket_count": 200
  },

  "validation_dataset_reader": {
      "type": "passage_attn2bio_reader",
      "joint_count": false,
      "count_samples_per_bucket_count": 100
  },

  "train_data_path": std.extVar("TRAIN_FILE"),
  "validation_data_path": std.extVar("VAL_FILE"),

  "model": {
      "type": "drop_pattn2bio",

      "joint_count": false,
      "passage_attention_to_span": {
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
      "patience": 15,
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
      "validation_metric": "+total_acc"
  },

  "random_seed": utils.parse_number(std.extVar("SEED")),
  "numpy_seed": utils.parse_number(std.extVar("SEED")),
  "pytorch_seed": utils.parse_number(std.extVar("SEED"))
}
