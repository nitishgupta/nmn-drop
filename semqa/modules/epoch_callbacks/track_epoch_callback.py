from typing import Dict, Any
from allennlp.training.trainer import EpochCallback, GradientDescentTrainer


@EpochCallback.register("track_epoch_callback")
class TrackEpochCallback:
    """
    A callback that you can pass to the `GradientDescentTrainer` to access the current epoch number in progress in the
    model as a class member `model.epoch`. Since the EpochCallback passes `epoch=-1` at the start of the training,
    we set `model.epoch = epoch + 1` which now denotes the number of completed epochs at a given training state.
    """
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_master: bool,
    ) -> None:
        trainer.model.epoch = epoch + 1
