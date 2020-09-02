# import os
# from allennlp.common.params import Params
# from allennlp.training import util as training_util
# from allennlp.models.model import Model
# from allennlp_models.generation.models.bart import Bart
# from allennlp.training.checkpointer import Checkpointer
# from allennlp.training.trainer import Trainer
# from allennlp.training.no_op_trainer import NoOpTrainer
#
#
# parameter_filename = "semqa/paired_data/bart_no_op.json"
# serialization_dir = "/shared/nitishg/checkpoints/bart_mlm"
#
# params = Params.from_file(parameter_filename)
# vocab = training_util.make_vocab_from_params(
#                 params.duplicate(), serialization_dir, print_statistics=False
#             )
#
# vocab_dir = os.path.join(serialization_dir, "vocabulary")
#
# params["vocabulary"] = {
#             "type": "from_files",
#             "directory": vocab_dir,
#             "padding_token": vocab._padding_token,
#             "oov_token": vocab._oov_token,
#         }
#
# # model = Model.load(params, serialization_dir)
# model = Bart(model_name="facebook/bart-large", vocab=vocab)
#
# trainer = NoOpTrainer(serialization_dir=serialization_dir, model=model)
#
# trainer.train()
#
# print("Bart model saved at: {}".format(serialization_dir))
