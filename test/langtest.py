from semqa.domain_languages.drop_language import get_empty_language_object


dl = get_empty_language_object()
# print(dl.all_possible_productions())

# diff - num num
# lf1 = "(passagenumber_difference (find_PassageNumber find_PassageAttention) (find_PassageNumber find_PassageAttention))"
#
# # count
# lf2 = "(passageAttn2Count find_PassageAttention)"
#
# # diff - min num
# lf3 = "(passagenumber_difference (min_PassageNumber (find_PassageNumber find_PassageAttention)) (min_PassageNumber (find_PassageNumber find_PassageAttention)))"

outer1 = "(find_passageSpanAnswer (relocate_PassageAttention "
outer2 = "))"
find = "find_PassageAttention"
filterlf = "(filter_PassageAttention find_PassageAttention)"
maxfind = "(maxNumPattn find_PassageAttention)"
maxfilterfind = f"(maxNumPattn {filterlf})"

gold_lf = outer1 + filterlf + outer2
action_seq = dl.logical_form_to_action_sequence(gold_lf)
print(action_seq)
print(len(action_seq))

print()

gold_lf = outer1 + find + outer2
action_seq = dl.logical_form_to_action_sequence(gold_lf)
print(action_seq)
print(len(action_seq))


print()

gold_lf = outer1 + maxfind + outer2
action_seq = dl.logical_form_to_action_sequence(gold_lf)
print(action_seq)
print(len(action_seq))

print()

gold_lf = outer1 + maxfilterfind + outer2
action_seq = dl.logical_form_to_action_sequence(gold_lf)
print(action_seq)
print(len(action_seq))
