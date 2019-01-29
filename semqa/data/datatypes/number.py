# pylint: disable=no-self-use
from typing import Any, Dict, List, Mapping

from overrides import overrides

from allennlp.data.fields.field import DataArray, Field
from allennlp.data.fields import MetadataField


class NumberField(MetadataField):
    """
    Class to store Data metadata.
    Parameters
    ----------
    date: List[int]
        [day, month, year]
    """
    def __init__(self, value: float) -> None:
        super().__init__(metadata=value)
        self.value = value

    @staticmethod
    def empty_object() -> 'NumberField':
        return NumberField(value=0.0)

    @overrides
    def empty_field(self) -> 'NumberField':
        return NumberField(value=0.0)

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]):  # -> DateField:
        # pylint: disable=unused-argument
        return self  # type: ignore

    def __str__(self) -> str:
        return f"NumberField with value: {self.value}."

