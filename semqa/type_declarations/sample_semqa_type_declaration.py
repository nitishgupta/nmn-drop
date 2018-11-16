from typing import List, Set

import allennlp

from allennlp.semparse.worlds.world import World
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

NUM_TYPE = NamedBasicType("NUM")
BOOL_TYPE = NamedBasicType("BOOLEAN")
QSTR_TYPE = NamedBasicType("QSTR")

BASIC_TYPES = {NUM_TYPE, BOOL_TYPE, QSTR_TYPE}
START_TYPES = {BOOL_TYPE}


''' First define all possible function signatures as complex types'''

# Functions: Return type is the last type
BOOL_FROM_ONENUM_TYPE = ComplexType(NUM_TYPE, BOOL_TYPE)
BOOL_FROM_TWONUM_TYPE = ComplexType(NUM_TYPE, ComplexType(NUM_TYPE, BOOL_TYPE))
BOOL_FROM_QSTR_TYPE = ComplexType(QSTR_TYPE, BOOL_TYPE)
NUM_FROM_TWONUM_TYPE = ComplexType(NUM_TYPE, ComplexType(NUM_TYPE, NUM_TYPE))
NUM_FROM_ONENUM_TYPE = ComplexType(NUM_TYPE, NUM_TYPE)


''' Now define all functions and map their names to a name the NLTK `LogicParser` understands using name_mapper '''
name_mapper = NameMapper()  # pylint: disable=invalid-name

# BOOL functions
name_mapper.map_name_with_signature("number_threshold", BOOL_FROM_ONENUM_TYPE)
name_mapper.map_name_with_signature("number_greater", BOOL_FROM_TWONUM_TYPE)
name_mapper.map_name_with_signature("ques_bool", BOOL_FROM_QSTR_TYPE)

# NUM Questions
name_mapper.map_name_with_signature("ground_num", NUM_TYPE)
name_mapper.map_name_with_signature("multiply", NUM_FROM_TWONUM_TYPE)
name_mapper.map_name_with_signature("scalar_mult", NUM_FROM_ONENUM_TYPE)



COMMON_NAME_MAPPING = name_mapper.common_name_mapping
COMMON_TYPE_SIGNATURE = name_mapper.common_type_signature

print("name mapping:")
print(COMMON_NAME_MAPPING)

print("type signature:")
print(COMMON_TYPE_SIGNATURE)
'''


world = World(global_type_signatures=COMMON_TYPE_SIGNATURE, global_name_mapping=COMMON_NAME_MAPPING)

va = types.get_valid_actions(name_mapping=world.get_name_mapping(), type_signatures=world.get_type_signatures(),
                             basic_types=BASIC_TYPES, valid_starting_types=START_TYPES,
                             num_nested_lambdas=0)

print(va)

for key, prods in va.items():

    for prod in prods:
        nt = types.is_nonterminal(prod)
        print(f"{prod}  {nt}")

print(types.is_nonterminal("n"))

'''