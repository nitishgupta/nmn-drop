from typing import List, Set

import allennlp

from allennlp.semparse import World
from overrides import overrides

from nltk.sem.logic import TRUTH_TYPE, BasicType, EntityType, Type

from allennlp.semparse.type_declarations import type_declaration as types

from allennlp.semparse.type_declarations.type_declaration import (ComplexType, HigherOrderType,
                                                                  NamedBasicType, NameMapper)

# This type declaration is based on: allennlp.semparse.type_declarations.nlvr_type_declaration

# All constants default to ``EntityType`` in NLTK. For domains where constants of different types
# appear in the logical forms, we have a way of specifying ``constant_type_prefixes`` and passing
# them to the constructor of ``World``. However, in the NLVR language we defined, we see constants
# of just one type, number. So we let them default to ``EntityType``.
# Commenting NUM_TYPE since our language does not contain constants
# NUM_TYPE = EntityType()

ENTITY_TYPE = NamedBasicType("ENTITY")
DATE_TYPE = NamedBasicType("DATE")
NUM_TYPE = NamedBasicType("NUM")
QSTRING_TYPE = NamedBasicType("QSTRING")
BOOL_TYPE = NamedBasicType("BOOLEAN")


''' First define possible function signatures as types'''
# A function that takes a QSTRING argument and returns an ENTITY_TYPE
SIMPLE_QA_ENTITY_TYPE = ComplexType(QSTRING_TYPE, ENTITY_TYPE)

# A function that takes a QSTRING argument and returns an ENTITY_TYPE
QA_ENTITYVAR_TYPE = ComplexType(QSTRING_TYPE, ComplexType(ENTITY_TYPE, ENTITY_TYPE))

# Functions to convert a pair of date/num into a boolean
BOOL_FROM_NUM_TYPE = ComplexType(NUM_TYPE, ComplexType(NUM_TYPE, BOOL_TYPE))
BOOL_FROM_DATE_TYPE = ComplexType(DATE_TYPE, ComplexType(DATE_TYPE, BOOL_TYPE))

BASIC_TYPES = {ENTITY_TYPE, DATE_TYPE, NUM_TYPE, QSTRING_TYPE, BOOL_TYPE}

''' Now define all functions and map their names to a name the NLTK `LogicParser` understands using name_mapper '''
name_mapper = NameMapper()  # pylint: disable=invalid-name

# Comparison functions
name_mapper.map_name_with_signature("number_equal", BOOL_FROM_NUM_TYPE)
name_mapper.map_name_with_signature("number_greater", BOOL_FROM_NUM_TYPE)
name_mapper.map_name_with_signature("number_lesser", BOOL_FROM_NUM_TYPE)

name_mapper.map_name_with_signature("date_equal", BOOL_FROM_DATE_TYPE)
name_mapper.map_name_with_signature("date_greater", BOOL_FROM_DATE_TYPE)
name_mapper.map_name_with_signature("date_lesser", BOOL_FROM_DATE_TYPE)

COMMON_NAME_MAPPING = name_mapper.common_name_mapping
COMMON_TYPE_SIGNATURE = name_mapper.common_type_signature


# print(COMMON_NAME_MAPPING)
# print(COMMON_TYPE_SIGNATURE)
#
# print(allennlp.__version__)
#
# world = World(global_name_mapping=COMMON_NAME_MAPPING, global_type_signatures=COMMON_TYPE_SIGNATURE)
#
# va = types.get_valid_actions(name_mapping=world.get_name_mapping(), type_signatures=world.get_type_signatures(),
#                              basic_types=BASIC_TYPES, valid_starting_types={BOOL_TYPE},
#                              num_nested_lambdas=0)
#
# world.all_possible_actions()
#
# print(va)
#
