import sys

''' This code takes two prediction files and compares how many predictions were corrected in the 2nd from the 1st
    File 1) Containing wrong (maybe also correct) predictions of a baseline model
    File 2) Containing predictions of the new model for which statistics will be computed. 
    
    Diclaimer: These two files can contain different questions, so only compute stats for overlapping questions
'''

# This contains the wrong predictions of a baseline model
baseline_predictions_file = sys.argv[1]
# This contains predictions from a new model
new_predications_file = sys.argv[2]


def getQuestionAnsPredAns(file_text):
    ''' Format of the file is txt, which contains Questions: Answer: Denotation: as fields. '''
    instances = file_text.split('\n\n')
    q_ans_preds = []
    for ins in instances:
        if ins.strip():
            tmptext = ins.split('Question: ')[1]
            ques, tmptext = tmptext.split('\n', maxsplit=1)
            tmptext = tmptext.split('Answer: ')[1]
            ans, tmptext = tmptext.split('\n', maxsplit=1)
            tmptext = tmptext.split('Denotation: ')[1]
            deno, tmptext = tmptext.split('\n', maxsplit=1)
            pred = 'no' if float(deno) < 0.5 else 'yes'
            correct = 'Correct' if ans == pred else 'Wrong'
            q_ans_preds.append((ques, ans, pred, correct, ins))

    return q_ans_preds


def main():
    with open(baseline_predictions_file, 'r') as f:
        baseline_predictions_text = f.read()

    with open(new_predications_file, 'r') as f:
        new_predictions_text = f.read()

    baseline_instances = getQuestionAnsPredAns(baseline_predictions_text)
    newmodel_instances = getQuestionAnsPredAns(new_predictions_text)

    ques_baseline = set([x[0] for x in baseline_instances])
    ques_newmodel = set([x[0] for x in newmodel_instances])

    common_questions = ques_baseline.intersection(ques_newmodel)
    print(common_questions)
    print(len(common_questions))

    for ins in newmodel_instances:
        if ins[0] in common_questions:
            print(ins)
            print()

if __name__=='__main__':
    main()
