from typing import List, Set

import allennlp

from allennlp.semparse.worlds.world import World
from overrides import overrides

from nltk.sem.logic import TRUTH_TYPE, BasicType, EntityType, Type

from allennlp.semparse.type_declarations import type_declaration as types

from allennlp.semparse.type_declarations.type_declaration import (ComplexType, HigherOrderType,
                                                                  NamedBasicType, NameMapper)

import datasets.hotpotqa.utils.constants as hpconstants

# This type declaration is based on: allennlp.semparse.type_declarations.nlvr_type_declaration

# All constants default to ``EntityType`` in NLTK. For domains where constants of different types
# appear in the logical forms, we have a way of specifying ``constant_type_prefixes`` and passing
# them to the constructor of ``World``. However, in the NLVR language we defined, we see constants
# of just one type, number. So we let them default to ``EntityType``.
# Commenting NUM_TYPE since our language does not contain constants
# NUM_TYPE = EntityType()

# NUM_TYPE = NamedBasicType("NUM")
# BOOL_TYPE = NamedBasicType("BOOLEAN")
# QSTR_TYPE = NamedBasicType("QSTR")

ANS_TYPES = [hpconstants.ENTITY_TYPE, hpconstants.NUM_TYPE, hpconstants.DATE_TYPE, hpconstants.BOOL_TYPE]


NUM_TYPE = NamedBasicType(hpconstants.NUM_TYPE)
BOOL_TYPE = NamedBasicType(hpconstants.BOOL_TYPE)
# ENT_TYPE = NamedBasicType(hpconstants.ENTITY_TYPE)
# STR_TYPE = NamedBasicType(hpconstants.STRING_TYPE)
# DATE_TYPE = NamedBasicType(hpconstants.DATE_TYPE)
QSTR_TYPE = NamedBasicType("QSTR")
QENT_TYPE = NamedBasicType("TQENT")

BASIC_TYPES = {NUM_TYPE, BOOL_TYPE, QSTR_TYPE, QENT_TYPE}
START_TYPES = {BOOL_TYPE, NUM_TYPE}


''' First define all possible function signatures as complex types'''

# Functions: Return type is the last type
BOOL_FROM_ONENUM_TYPE = ComplexType(NUM_TYPE, BOOL_TYPE)
BOOL_FROM_TWONUM_TYPE = ComplexType(NUM_TYPE, ComplexType(NUM_TYPE, BOOL_TYPE))
NUM_FROM_TWONUM_TYPE = ComplexType(NUM_TYPE, ComplexType(NUM_TYPE, NUM_TYPE))
NUM_FROM_ONENUM_TYPE = ComplexType(NUM_TYPE, NUM_TYPE)

BOOL_FROM_QSTR_TYPE = ComplexType(QSTR_TYPE, BOOL_TYPE)
BOOL_FROM_QENT_TYPE = ComplexType(QENT_TYPE, BOOL_TYPE)
BOOL_FROM_2QSTR_TYPE = ComplexType(QSTR_TYPE, ComplexType(QSTR_TYPE, BOOL_TYPE))



''' Now define all functions and map their names to a name the NLTK `LogicParser` understands using name_mapper '''
name_mapper = NameMapper()  # pylint: disable=invalid-name

# BOOL functions
name_mapper.map_name_with_signature("number_threshold", BOOL_FROM_ONENUM_TYPE)
name_mapper.map_name_with_signature("number_greater", BOOL_FROM_TWONUM_TYPE)

name_mapper.map_name_with_signature("ques_bool", BOOL_FROM_QSTR_TYPE)
name_mapper.map_name_with_signature("ques_ent_bool", BOOL_FROM_QENT_TYPE)
name_mapper.map_name_with_signature("two_ques_bool", BOOL_FROM_2QSTR_TYPE)


# NUM Questions
name_mapper.map_name_with_signature("ground_num", NUM_TYPE)
name_mapper.map_name_with_signature("multiply", NUM_FROM_TWONUM_TYPE)
name_mapper.map_name_with_signature("scalar_mult", NUM_FROM_ONENUM_TYPE)



COMMON_NAME_MAPPING = name_mapper.common_name_mapping
COMMON_TYPE_SIGNATURE = name_mapper.common_type_signature

# These are all function-types that take more than one argument. Needed for world._get_curried_functions
curried_functions = {
                         BOOL_FROM_TWONUM_TYPE: 2,
                         BOOL_FROM_2QSTR_TYPE: 2,
                         NUM_FROM_TWONUM_TYPE: 2,
                     }

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