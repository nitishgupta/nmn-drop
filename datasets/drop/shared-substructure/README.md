# Shared-substructure data augmentation

Idea is to augment questions with additional supervision of the following kind; w/ questions whose program shares a 
common sub-tree with the original question's program. Given such annotation we can encourage the model to predict the 
same output for the two sub-trees. Since the representations for the question, passage and the  subsequent token2num & 
token2date alignments would be conditioned on different questions, this should encourage the model to learn the produce 
representations that are independent (or not too dependent) on the other parts of the question.

Example,
given question: "Who kicked the longest field goal?" with the program: 
`project["who kicked"](find-max-num find["field goal"])`

We can supervise with question: "How long was the shortest field goal?" with the program: 
`select-num(find-min-num find["field goal"])`, 

that the outputs of the `find["field goal"]` nodes in the two programs should be the same.

This should encourage the model to learn "longest" and "shortest" independent `question` and `passage` representations.


### Annotation guidiline
We add this annotation in a new `shared_substructure_annotations` key in the qa-dict in the DROP data.
Since each question can be supervised with multiple questions, this key contains a `List[Dict]` where each `Dict` is a 
separate annotation with the following keys:
```
# Shared-substructure augmented question example
{
    constants.question: str = e.g. "How long was the shortest field goal?",
    constants.question_tokens: List[str] = ["How", "long", ...],
    constants.program_supervision: Dict = aux_program_node.to_dict(),
    "orig_program_lisp": str = e.g. "project(find-max-num find)",
    "orig_question": str = "Who kicked the longest field goal?",
    "origprog_postorder_node_idx": int = Node-idx in the original program (post-order indexing) for the common node ,
    "sharedprog_postorder_node_idx": int = Node-idx in the aux program (post-order indexing) for the common node
}
```
`origprog_postorder_node_idx` and `sharedprog_postorder_node_idx` are the indices of the common node in the original 
and the augmented program when indexing in a post-order traversal. 
E.g. the `find` node in the two programs above would be `idx=0` in both the orig. and the aux. program.
