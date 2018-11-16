"""
This module defines classes Object and Box (the two entities in the NLVR domain) and an NlvrWorld,
which mainly contains an execution method and related helper methods.
"""
from typing import Dict, Set, List
import logging
import sys

from nltk.sem.logic import Type
from overrides import overrides

from semqa.type_declarations import sample_semqa_type_declaration as types
from allennlp.semparse.worlds.world import ParsingError, World
from semqa.executors.hotpotqa.sample_executor import SampleHotpotExecutor

from allennlp.semparse.worlds.wikitables_world import WikiTablesWorld
from allennlp.semparse.worlds.atis_world import AtisWorld

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
    curried_functions = {
            types.BOOL_FROM_TWONUM_TYPE: 2
            }

    def __init__(self, ques_spans: List[str]=None) -> None:
        super(SampleHotpotWorld, self).__init__(global_type_signatures=types.COMMON_TYPE_SIGNATURE,
                                                global_name_mapping=types.COMMON_NAME_MAPPING,
                                                num_nested_lambdas=0)
        """
        Parameters:
        -----------
        ques_spans: List of question spans (tokens delimited by _) to be used as actions
        """

        # boxes = set([Box(object_list, box_id) for box_id, object_list in
        #              enumerate(world_representation)])

        if ques_spans is not None:
            for ques_str in ques_spans:
                ques_str = f"qstr:{ques_str}"
                print(ques_str)
                self._map_name2(name=ques_str, keep_mapping=True)
                # self.print_name(name=ques_str)
                print("finished")


        print(self.get_name_mapping())
        print(self.get_type_signatures())

        print(self.all_possible_actions())

        sys.exit()



        self._executor = SampleHotpotExecutor()

        # print(f"Name Mapping Signature: {types.COMMON_NAME_MAPPING}")
        # print(f"Type Signature: {types.COMMON_TYPE_SIGNATURE}")
        #
        #
        # print("Local Name Mapping")
        # print(self.local_name_mapping)
        # print(self.local_type_signatures)

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

    # @overrides
    def _map_name2(self, name: str, keep_mapping: bool = False) -> str:
        print(name)
        if name not in types.COMMON_NAME_MAPPING and name not in self.local_name_mapping:
            if not keep_mapping:
                raise ParsingError(f"Encountered un-mapped name: {name}")
            print(name)
            if name.startswith("qstr:"):
                # Question sub-span
                print("HERE")
                translated_name = "Q:" + name
                self._add_name_mapping(name, translated_name, types.QSTR_TYPE)
            else:
                raise ParsingError(f"Cannot handle Names apart from qstr:ques_tokens. Input: {name}")
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

    @overrides
    def _map_name(self, name: str, keep_mapping: bool = False) -> str:
        return types.COMMON_NAME_MAPPING[name] if name in types.COMMON_NAME_MAPPING else name

    def execute(self, logical_form: str) -> bool:
        return self._executor.execute(logical_form)