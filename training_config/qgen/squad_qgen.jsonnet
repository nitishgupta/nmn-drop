local utils = import 'utils.libsonnet';

local bert_model="facebook/bart-large";
local beam_size = utils.parse_number(std.extVar("BEAMSIZE"));
local seed = utils.parse_number(std.extVar("SEED"));
local batch_size = utils.parse_number(std.extVar("BS"));
local masked_question = utils.boolparser(std.extVar("MASKQ"));

{
  "dataset_reader": {
      "type": "squad_conditional_qgen",
      "model_name": bert_model,
      "add_masked_question": masked_question,
      "lazy": false
  },

  "train_data_path": std.extVar("TRAIN_DATA"),
  "validation_data_path": std.extVar("VAL_DATA"),

  "model": {
      "type": "conditional_qgen",
      "model_name": bert_model,
      "beam_size": beam_size,
  },

//  "data_loader": {
//    "batch_sampler": {
//      "type": "bucket",
//      "batch_size": 8
//    },
//  },

  "data_loader": {
      "batch_sampler": {
          "type": "basic",
          "sampler": {"type": "random"},
          "batch_size": batch_size,
          "drop_last": false,
      },
  },

  "trainer": {
      "checkpointer": {
        "num_serialized_models_to_keep": 1
      },
      "num_epochs": utils.parse_number(std.extVar("EPOCHS")),
      "cuda_device": utils.parse_number(std.extVar("GPU")),
      "grad_norm": 1,
      "optimizer": {
        "type": "huggingface_adamw",
        "lr": 3e-5,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "correct_bias": true
      }
  },
  "random_seed": seed,
  "numpy_seed": seed,
  "pytorch_seed": seed
}