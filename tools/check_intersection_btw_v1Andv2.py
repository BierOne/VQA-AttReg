import os, json
import itertools
from tqdm import tqdm

with open('/home/share/liuyibing/vqa/vqa-cp1.0/qa_path/vqacp_v1_train_questions.json', 'r') as fd:
    v1_questions_json = json.load(fd)
with open('/home/share/liuyibing/vqa/vqa-cp1.0/qa_path/vqacp_v1_test_questions.json', 'r') as fd:
    v1_questions_json2 = json.load(fd)
v1_questions_json = itertools.chain(v1_questions_json, v1_questions_json2)
with open('/home/share/liuyibing/vqa/vqa-cp2.0/qa_path/vqacp_v2_train_questions.json', 'r') as fd:
    v2_questions_json = json.load(fd)
with open('/home/share/liuyibing/vqa/vqa-cp2.0/qa_path/vqacp_v2_test_questions.json', 'r') as fd:
    v2_questions_json2 = json.load(fd)
v2_questions_json = itertools.chain(v2_questions_json, v2_questions_json2)

v1_question_ids = [q['question_id'] for q in v1_questions_json]
v2_question_ids = [q['question_id'] for q in v2_questions_json]

intersection_num = 0
for qid in tqdm(v1_question_ids):
    if qid in v2_question_ids:
        intersection_num += 1
print("intersection_num/v1_num: {:d}/{:d}, intersection_num/v2_num: {:d}/{:d}"
              .format(intersection_num, len(v1_question_ids), intersection_num, len(v2_question_ids)) )