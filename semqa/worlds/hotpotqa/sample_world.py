"""
This module defines classes Object and Box (the two entities in the NLVR domain) and an NlvrWorld,
which mainly contains an execution method and related helper methods.
"""
from typing import Dict, Set, List
import logging
import sys

from nltk.sem.logic import Type
from overrides import overrides

from allennlp.semparse.worlds.world import ParsingError, World
from semqa.executors.hotpotqa.sample_executor import SampleHotpotExecutor

# from semqa.type_declarations import sample_semqa_type_declaration as types
from semqa.type_declarations import semqa_type_declaration_wques as types

from allennlp.semparse.worlds.wikitables_world import WikiTablesWorld
from allennlp.semparse.worlds.atis_world import AtisWorld

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

QSTR_PREFIX="QSTR:"


class SampleHotpotWorld(World):
    """
    Copied from the NLVRWorld (https://github.com/allenai/allennlp/blob/master/allennlp/semparse/worlds/nlvr_world.py)
    Class defining the world representation of NLVR. Defines an execution logic for logical forms
    in NLVR.  We just take the structured_rep from the JSON file to initialize this.

    Parameters
    ----------
    world_representation : ``JsonDict``
        structured_rep from the JSON file.
    """
    # pylint: disable=too-many-public-methods

    # When we're converting from logical forms to action sequences, this set tells us which
    # functions in the logical form are curried functions, and how many arguments the function
    # actually takes.  This is necessary because NLTK curries all multi-argument functions to a
    # series of one-argument function applications.  See `world._get_transitions` for more info.
    curried_functions = types.curried_functions

    def __init__(self, ques_spans: List[str]=None) -> None:
        super(SampleHotpotWorld, self).__init__(global_type_signatures=types.COMMON_TYPE_SIGNATURE,
                                                global_name_mapping=types.COMMON_NAME_MAPPING,
                                                num_nested_lambdas=0)
        """
        Parameters:
        -----------
        ques_spans: List of question spans (tokens delimited by _) to be used as actions
        """


        if ques_spans is not None:
            for ques_str in ques_spans:
                ques_str = f"{QSTR_PREFIX}{ques_str}"
                self._map_name(name=ques_str, keep_mapping=True)


        self._executor = SampleHotpotExecutor()

        ''' These terminal productions are used for agenda in NLVR. Shouldn't need in regular cases '''
        """
        # Mapping from terminal strings to productions that produce them.
        # Eg.: "yellow" -> "<o,o> -> yellow", "<b,<<b,e>,<e,b>>> -> filter_greater" etc.
        self.terminal_productions: Dict[str, str] = {}
        for constant in types.COMMON_NAME_MAPPING:
            alias = types.COMMON_NAME_MAPPING[constant]
            if alias in types.COMMON_TYPE_SIGNATURE:
                constant_type = types.COMMON_TYPE_SIGNATURE[alias]
                self.terminal_productions[constant] = "%s -> %s" % (constant_type, constant)
        """

    def print_name(self, name):
        print(name)

    @overrides
    def _map_name(self, name: str, keep_mapping: bool = False) -> str:
        """
        Naming convention for translated name:
            - Function variables: Single upper case letter optionally followed by digits
            - Individual variables: Single lower case letter (except e for events) optionally
              followed by digits
            - Constants: Everything else


        _add_name_mapping -- Adds the name: translated_name mapping in the name_mapping dict
            and adds the translated_name: Type mapping in the type_signature dict.
            Eg. 'number_threshold' (name) : 'F0' (translated_name) in the name_mapping and
                'F0':  <n,b> (Type) in the type_signature.

        Since instance-specific actions we add are terminals, translated name can be the same as the name
        'Type -> name' will be added as the action. Translated_name convention is only used for internal purposes
        """
        if name not in types.COMMON_NAME_MAPPING and name not in self.local_name_mapping:
            if not keep_mapping:
                raise ParsingError(f"Encountered un-mapped name: {name}")
            if name.startswith(QSTR_PREFIX):
                # Question sub-span
                translated_name = name
                self._add_name_mapping(name, translated_name, types.QSTR_TYPE)
            else:
                raise ParsingError(f"Cannot handle names apart from qstr:ques_tokens. Input: {name}")
        else:
            if name in types.COMMON_NAME_MAPPING:
                translated_name = types.COMMON_NAME_MAPPING[name]
            else:
                translated_name = self.local_name_mapping[name]
        return translated_name

    @overrides
    def get_basic_types(self) -> Set[Type]:
        return types.BASIC_TYPES

    @overrides
    def get_valid_starting_types(self) -> Set[Type]:
        return types.START_TYPES


    @overrides
    def get_valid_actions(self) -> Dict[str, List[str]]:
        valid_actions = super().get_valid_actions()
        return valid_actions

    def _get_curried_functions(self) -> Dict[Type, int]:
        return SampleHotpotWorld.curried_functions

    def execute(self, logical_form: str) -> bool:
        return self._executor.execute(logical_form)