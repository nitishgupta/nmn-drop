local bert_model="facebook/bart-large";

{
  "dataset_reader": {
    "type": "question_generation",
    "model_name": bert_model,
    "lazy": false
  },
  "train_data_path": "/shared/nitishg/data/qgen_dand/train.jsonl",
  "validation_data_path": "/shared/nitishg/data/qgen_dand/valid.jsonl",
  "model": {
    "type": "question_generation",
    "model_name": bert_model,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 8
    },
  },
  "trainer": {
    "checkpointer": {
      "num_serialized_models_to_keep": 1
    },
    "num_epochs": 1,
    "cuda_device": 0,
    "grad_norm": 1,
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 3e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "correct_bias": true
    }
  },
  "random_seed": 4,
  "numpy_seed": 5,
  "pytorch_seed": 6
}