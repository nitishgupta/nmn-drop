import os
import json

predictions_json = "/shared/nitishg/data/drop_acl/nerd-preds/predictions.json"
nbest_preds_json = "/shared/nitishg/data/drop_acl/nerd-preds/nbest_predictions.json"

hmyw_dev_json = "/shared/nitishg/data/drop_acl/num/how_many_yards_was/drop_dataset_dev.json"
whoarg_dev_json = "/shared/nitishg/data/drop_acl/num/who_arg/drop_dataset_dev.json"
dev_dataset_json = "/shared/nitishg/data/drop_acl/raw/drop_dataset_dev.json"


def read_qid2ques_and_passage(dataset):
    qid2ques = {}
    qid2passage = {}
    for pid, pinfo in dataset.items():
        passage = pinfo["passage"]
        for qa_pair in pinfo["qa_pairs"]:
            qid = qa_pair["query_id"]
            question = qa_pair["question"]
            qid2ques[qid] = question
            qid2passage[qid] = passage
            # if "longest" in question or "shortest" in question:
            #     minmax_qid2ques[qid] = question

    return qid2ques, qid2passage


def get_minmax_ques(dataset):
    minmax_qid2ques = {}
    for pid, pinfo in dataset.items():
        for qa_pair in pinfo["qa_pairs"]:
            qid = qa_pair["query_id"]
            question = qa_pair["question"]
            if "longest" in question or "shortest" in question:
                minmax_qid2ques[qid] = question
    return minmax_qid2ques


def count_min_max(pred_programs):
    num_min_max_progs = 0
    total_values_in_min_max = 0
    max_values_in_prog = 0
    for qid, program in pred_programs.items():
        if "MIN(" in program or "MAX(" in program:
            num_min_max_progs += 1
            num_values = program.count("VALUE(")
            total_values_in_min_max += num_values
            max_values_in_prog = max(max_values_in_prog, num_values)

    avg_num_values = float(total_values_in_min_max)/num_min_max_progs
    print(f"Num min/max progs: {num_min_max_progs}")
    print(f"Avg num values: {avg_num_values}")
    print(f"max num values: {max_values_in_prog}")


def count_arg_min_max(pred_programs):
    num_min_max_progs = 0
    total_values_in_min_max = 0
    max_values_in_prog = 0
    argminmax_qids = []
    for qid, program in pred_programs.items():
        if "ARGMIN(" in program or "ARGMAX(" in program:
            num_min_max_progs += 1
            num_values = program.count("KV(")
            total_values_in_min_max += num_values
            max_values_in_prog = max(max_values_in_prog, num_values)
            argminmax_qids.append(qid)

    avg_num_values = float(total_values_in_min_max)/num_min_max_progs
    print(f"Num ARG min/max progs: {num_min_max_progs}")
    print(f"Avg num key-values: {avg_num_values}")
    print(f"max num key-values: {max_values_in_prog}")
    return argminmax_qids


def is_minmax_prog(program):
    if "MIN(" in program or "MAX(" in program or "ARGMIN(" in program or "ARGMAX(" in program:
        return True
    else:
        return False


def num_of_minmaxpred(minmax_qid2ques, pred_programs, dev_qid2passage, predictions, nbest_preds):
    total_qa = len(minmax_qid2ques)
    total_minmax_predprogram = 0
    qid_not_minmax = []

    correct_minmaxpred = 0
    correct_not_minmaxpred = 0

    for qid, question in minmax_qid2ques.items():
        predprogram = pred_programs[qid]
        isminmax = is_minmax_prog(predprogram)
        pred_ans = predictions[qid]
        ref_answers = nbest_preds[qid][0]["ref_answer"]
        iscorrect = is_correct_func(pred_ans, ref_answers)
        # if isminmax:
        #     print("{}\n{}\n{}\n".format(question, predprogram, dev_qid2passage[qid]))

        total_minmax_predprogram += 1 if isminmax else 0
        correct_minmaxpred += int(iscorrect) if isminmax else 0
        correct_not_minmaxpred += int(iscorrect) if not isminmax else 0
        if not isminmax:
            qid_not_minmax.append(qid)

    print("Total min/max questions : {}".format(total_qa))
    print("Pred program are min/max: : {}".format(total_minmax_predprogram))
    print("Correct amongst min/max pred: : {}".format(correct_minmaxpred))
    print("Correct amongst NON min/max pred: : {}".format(correct_not_minmaxpred))
    return qid_not_minmax


def is_correct_func(pred, refs):
    pred = [str(x) for x in pred]
    for ref in refs:
        if ref == pred:
            return True

    return False


def write_not_minmax_to_file(txt_outfile, qid_not_minmax, predictions, pred_programs, dev_qid2ques, dev_qid2passage,
                             nbest_preds):
    with open(txt_outfile, 'w') as outf:
        for qid in qid_not_minmax:
            question = dev_qid2ques[qid]
            passage = dev_qid2passage[qid]
            predprogram = pred_programs[qid]
            pred_ans = predictions[qid]
            ref_answers = nbest_preds[qid][0]["ref_answer"]

            outf.write("{}\n{}\n{}\npred:{}\nref:{}\n\n".format(question, predprogram, passage, pred_ans, ref_answers))


def write_incorrect_to_txtfile(txt_outfile, predictions, pred_programs, dev_qid2ques, dev_qid2passage,
                               nbest_preds):
    print("\nWriting incorrect dev predictions ... ")
    total_ques_written = 0
    total_ques = len(dev_qid2ques)
    with open(txt_outfile, 'w') as outf:
        for qid, question in dev_qid2ques.items():
            passage = dev_qid2passage[qid]
            predprogram = pred_programs[qid]
            pred_ans = predictions[qid]
            ref_answers = nbest_preds[qid][0]["ref_answer"]
            iscorrect = is_correct_func(pred_ans, ref_answers)
            if iscorrect:
                continue
            outf.write("{}\n{}\n{}\npred:{}\nref:{}\n\n".format(question, passage, predprogram, pred_ans, ref_answers))
            total_ques_written += 1
    print("Total: {} Incorrect:{}".format(total_ques, total_ques_written))


def write_incorrect_to_tsvfile(txt_outfile, predictions, pred_programs, dev_qid2ques, dev_qid2passage,
                               nbest_preds):
    print("\nWriting incorrect dev predictions ... ")
    total_ques_written = 0
    total_ques = len(dev_qid2ques)
    with open(txt_outfile, 'w') as outf:
        outf.write(f"Question\tPredProgram\tPredAns\tGoldAns\tPassage\n")
        for qid, question in dev_qid2ques.items():
            passage = dev_qid2passage[qid]
            predprogram = pred_programs[qid]
            pred_ans = predictions[qid]
            ref_answers = nbest_preds[qid][0]["ref_answer"]
            iscorrect = is_correct_func(pred_ans, ref_answers)
            if iscorrect:
                continue
            outf.write(f"{question}\t{predprogram}\t{pred_ans}\t{ref_answers}\t{passage}\n")
            total_ques_written += 1
    print("Total: {} Incorrect:{}".format(total_ques, total_ques_written))



def main():
    hmyw_dev = json.load(open(hmyw_dev_json))
    whoarg_dev = json.load(open(whoarg_dev_json))
    dev_dataset = json.load(open(dev_dataset_json))
    # Dict from qid to List containing predictions
    predictions = json.load(open(predictions_json))
    nbest_preds = json.load(open(nbest_preds_json))
    # Dict from qid to List[program_tokens]
    pred_programs = predictions["pred_programs"]


    dev_qid2ques, dev_qid2passage = read_qid2ques_and_passage(dev_dataset)

    whoarg_minmax_qid2ques = get_minmax_ques(whoarg_dev)
    hmyw_minmax_qid2ques = get_minmax_ques(hmyw_dev)


    count_min_max(pred_programs)
    argminmax_qids = count_arg_min_max(pred_programs)

    argminmax_not_whoarg = set(argminmax_qids).difference(set(whoarg_minmax_qid2ques.keys()))
    print([dev_qid2ques[qid] for qid in argminmax_not_whoarg])

    print("HMYW")
    hmyw_not_minmaxpreds = num_of_minmaxpred(hmyw_minmax_qid2ques, pred_programs, dev_qid2passage,
                                             predictions, nbest_preds)
    print("Who Arg")
    whoarg_not_minmaxpreds = num_of_minmaxpred(whoarg_minmax_qid2ques, pred_programs, dev_qid2passage,
                                               predictions, nbest_preds)

    print("\nWriting incorrect min/max HMYW dev predictions ... ")
    write_not_minmax_to_file(txt_outfile="/shared/nitishg/data/drop_acl/nerd-preds/hmwy_notminmax.json",
                             qid_not_minmax=hmyw_not_minmaxpreds,
                             predictions=predictions, pred_programs=pred_programs,
                             dev_qid2ques=dev_qid2ques, dev_qid2passage=dev_qid2passage, nbest_preds=nbest_preds)

    print("\nWriting incorrect min/max WHO-ARG dev predictions ... ")
    write_not_minmax_to_file(txt_outfile="/shared/nitishg/data/drop_acl/nerd-preds/whoarg_notminmax.json",
                             qid_not_minmax=whoarg_not_minmaxpreds,
                             predictions=predictions, pred_programs=pred_programs,
                             dev_qid2ques=dev_qid2ques, dev_qid2passage=dev_qid2passage, nbest_preds=nbest_preds)

    # write_incorrect_to_txtfile
    write_incorrect_to_tsvfile(txt_outfile="/shared/nitishg/data/drop_acl/nerd-preds/dev_incorrect.tsv",
                               predictions=predictions, pred_programs=pred_programs, dev_qid2ques=dev_qid2ques,
                               dev_qid2passage=dev_qid2passage, nbest_preds=nbest_preds)


if __name__ == '__main__':
    main()


