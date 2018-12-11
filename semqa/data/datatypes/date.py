# pylint: disable=no-self-use
from typing import Any, Dict, List, Mapping

from overrides import overrides

from allennlp.data.fields.field import DataArray, Field
from allennlp.data.fields import MetadataField


class DateField(MetadataField):
    """
    Class to store Data metadata.
    Parameters
    ----------
    date: List[int]
        [day, month, year]
    """
    def __init__(self, date: List[int]) -> None:
        assert len(date) == 3, "Need a list of size 3 to initialize date"
        super().__init__(metadata=date)
        self.day = date[0]
        self.month = date[1]
        self.year= date[2]

    @staticmethod
    def empty_object() -> 'DateField':
        return DateField(date=[-1, -1, -1])

    @overrides
    def empty_field(self) -> 'DateField':
        return DateField(date=[-1, -1, -1])

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]): #-> DateField:
        # pylint: disable=unused-argument
        return self # type: ignore

    def __str__(self) -> str:
        return f"DateField with date: ({self.day}, {self.month}, {self.year})."




