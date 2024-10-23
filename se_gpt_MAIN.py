# -*- coding: utf-8 -*-

"""
Dear Readers,

Thank you for your attention to our work!

We plan to submit a progress paper on improving SE-GPT to ACL ARR December 2024.
This progress paper primarily addresses the issues of se-gpt related to high time 
and financial costs and the lack of adaptability to various task types. 
Due to the code reuse and overlapping research paths of both projects, 
coupled with the intense competition in the current NLP field, 
we will delay releasing the complete code until after the paper notification.
We hope for your understanding. 

To minimize the inconvenience caused, 
we share some experiences below related to experiential learning. 
In addition, we provide this code file, a preview that contains the main framework but lacks some details, 
which can make it easier for you to code for your own tasks. 
Of course, you are also welcome to email me (jlgao@ir.hit.edu.cn) to 
discuss the application of experiences to tasks of your interest (if our research directions do not conflict).
 More in-depth paper collaborations (co-authorship) are also expected.

The concept of textual experience is gradually becoming a popular topic 
at conferences such as EMNLP 2023 (four main papers and two findings papers) and AAAI.
Personally, I believe this insight can lead to interesting research in various tasks, such as (to my knowledge) autonomous
experience accumulation guiding fake event detection and attacks/defenses of large models.
If you are interested in exploring this insight in your own field and wish to conduct related research
in the short term, here are some practical suggestions:

- Sug: When summarizing experiences, it is more important to input examples where the large model made mistakes. Typically, examples where the large model was correct offer less benefit (this is also why SE-GPT stopped learning from such examples).

- Sug: The use of experiences can be very complex. Although SE-GPT directly adds experiences to the prompt due to the similarity of our experimental tasks, which all belong to text reasoning and classification, in our ongoing work, we find that trying to adapt the experience to the current specific question before using it may be a more general approach.

Additionally, due to time constraints, the following works were not included in the related work section of our paper:

- https://aclanthology.org/2023.emnlp-main.109/
- https://aclanthology.org/2023.emnlp-main.150/
- https://aclanthology.org/2023.emnlp-main.659/
- https://aclanthology.org/2023.emnlp-main.950/



Sincerely
"""
assert False


import json
import pdb
import random
import re
import traceback
from argparse import ArgumentParser

import tiktoken
import torch
from datasets import Dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer


def find_count_max(data):
    assert type(data) == list
    count = {}
    for d in data:
        if d not in count:
            count[d] = 0
        count[d] += 1
    max_count = 0
    max_d = None
    for d in count:
        if count[d] > max_count:
            max_count = count[d]
            max_d = d
    return max_d


parser = ArgumentParser()
parser.add_argument('--organization', type=str, default="")
parser.add_argument('--api_key', type=str)
parser.add_argument('--input_data_path', type=str)
parser.add_argument('--model_type', type=str)
args = parser.parse_args()

# set openai
if args.organization!="":
    client = OpenAI(
        organization=args.organization,
        api_key=args.api_key,
        max_retries=10,
        timeout=600
    )
else:
    client = OpenAI(
        api_key=args.api_key,
        max_retries=10,
        timeout=600
    )


def chat_create(model, max_tokens, seed, temperature, messages, log_flag="chat_create"):
    inputs = {"model": model, "max_tokens": max_tokens, "seed": seed, "temperature": temperature, "messages": messages}
    response = client.chat.completions.create(**inputs)
    json_response = json.loads(response.model_dump_json())
    response_content = json_response["choices"][0]["message"]["content"]
    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> start %s >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n" % log_flag
          + messages[0]["content"]
          + "\n" + "v" * 100 + "\n"
          + response_content
          + "\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end %s <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n" % log_flag)
    return response_content


# main todo

corpus = [json.loads(line) for line in open(args.input_data_path, 'r', encoding="utf-8").readlines()]

exp_data_recorder = open(args.input_data_path.strip() + ".exp_data.jsonl", 'w+', encoding="utf-8")

random.seed(0)

# faiss config
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
device = "cpu"
torch.set_grad_enabled(False)
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_encoder.to(device)

# build memory
TASK_MEMORY = None

# start evolving loop
for item_id, item in tqdm(enumerate(corpus), desc="MAIN", total=len(corpus)):
    print("######################### START %d #########################" % item_id)

    prompt = item["prompt"].strip()

    final_exp = None
    # < init name description
    print("1.1 init task name and description")
    if "task_type_induction" not in item:
        task_type_induction_prompt = """You are an advanced task type induction agent capable of naming a task and describing its goals based on an example of the task.
The description of the task goals should be abstract, general, and essential, avoiding any specifics about how the problem is described or the variable elements within it, as the same task can be described in various ways.
Use the following JSON format to output task name and task descriptions:
```json
{
  "task name": ,
  "task description":
}
```
< Task Example >
%s
</ Task Example >

""" % prompt

        # gpt-3.5-turbo-1106 gpt-4-1106-preview
        # for model_type in ("gpt-3.5-turbo-1106","gpt-4-1106-preview"):
        max_retry = 10
        for i in range(max_retry + 1):
            cur_res = None
            if i == max_retry:
                raise Warning("retry too many times")
            try:
                #
                model_type = args.model_type
                response = chat_create(model=model_type, max_tokens=512, seed=1 + i, temperature=1,
                                       messages=[{"role": "user", "content": task_type_induction_prompt}], log_flag="task_type_induction_prompt %d" % i)
                res = re.findall("```json(.*?)```", response, flags=re.S)
                if len(res) == 0:
                    res = re.findall(r"(\{.*?})", response, flags=re.S)
                tmp_out = json.loads(res[0])
                assert "task name" in tmp_out and "task description" in tmp_out
                cur_res = {"task name": tmp_out["task name"].strip(), "task description": tmp_out["task description"].strip()}
                break
            except Exception as e:
                print("retry %d/%d", i + 1, max_retry)
                traceback.print_exc()
                cur_res = None
        assert cur_res is not None
        item["task_type_induction"] = cur_res
        # end generate tti
    task_name = item["task_type_induction"]["task name"].strip()
    task_description = item["task_type_induction"]["task description"].strip()
    # /> init name description

    print("1.2 if visited, get eq task id from memory")
    # < select eq_src_id
    if TASK_MEMORY is None:
        print("1.2.1 no memory, skip 1.2")
        # < skip if no memory
        eq_src_id = -1
        # /> skip if no memory
    else:

        # < select from memory
        if len(TASK_MEMORY) == 1:
            top_indices = [0]
        else:
            question_embedding = q_encoder(**q_tokenizer(task_description, return_tensors="pt", padding=True, truncation=True, max_length=512))[0][
                0].numpy()
            _, indices = TASK_MEMORY.search('embeddings', question_embedding, k=10)  # top5 for tti, top10 for transfer
            top_indices = [int(i) for i in indices if i >= 0]
            # scores, retrieved_examples = scores[: len(top_indices)], TASK_MEMORY[top_indices]
        print("1.2.2 has memory, preliminary screening = %s" % str(top_indices))

        tmp_info = "<Target Task>\n%s\n</Target Task>" % task_description
        for order_id, selected_ind in enumerate(top_indices[:5]):
            tmp_info += "\n\n<Candidate Task %d>\n%s\n</Candidate Task %d>" \
                        % (order_id + 1,
                           TASK_MEMORY[selected_ind]["task_description"],
                           order_id + 1)
        is_eq_task_prompt = """%s

You are an excellent task identifier, capable of determining whether the target task is identical to one of the above candidate tasks.
If no such candidate tasks exist, or if you are unsure, please return -1.
You must carefully avoid selecting any candidate task that are not completely identical to the target task.
Please use the following JSON format to output the selected candidate task:
```json
{
"selected task id": /* -1 or ID of the selected candidate task. */
}
```""" % tmp_info
        # gpt-3.5-turbo-1106 gpt-4-1106-preview
        # for model_type in ("gpt-3.5-turbo-1106","gpt-4-1106-preview"):

        vote_log = []
        for vote_id in range(order_id + 3):
            if len(vote_log) != len(set(vote_log)):
                break
            max_retry = 5
            for i in range(max_retry + 1):
                if i == max_retry:
                    raise Warning("retry too many times")
                try:
                    #
                    model_type = args.model_type
                    response = chat_create(model=model_type, max_tokens=128, seed=1 + i + 1000 * vote_id, temperature=1,
                                           messages=[{"role": "user", "content": is_eq_task_prompt}], log_flag="is_eq_task_prompt %d" % i)
                    res = re.findall("```json(.*?)```", response, flags=re.S)
                    if len(res) == 0:
                        res = re.findall(r"(\{.*?})", response, flags=re.S)
                    tmp_out = json.loads(res[0])
                    assert "selected task id" in tmp_out
                    eq_src_id = int(tmp_out["selected task id"])
                    assert -1 <= eq_src_id <= min(5, len(top_indices))
                    vote_log.append(eq_src_id)
                    break
                except Exception as e:
                    print("retry %d/%d", i + 1, max_retry)
                    traceback.print_exc()
        assert len(vote_log) != len(set(vote_log))
        eq_src_id = find_count_max(vote_log)
        print("vote result = %d" % eq_src_id)
        print("1.2.3 has memory, detailed screening = %s" % str(eq_src_id))
        # /> select from memory
    print("1.2 end with eq_src_id = %s" % str(eq_src_id))
    # >/ select eq_src_id

    # if visited, reuse task_name, task_description and experience
    if eq_src_id != -1:
        eq_root_id = top_indices[eq_src_id - 1]
        task_name = TASK_MEMORY[eq_root_id]["task_name"]
        task_description = TASK_MEMORY[eq_root_id]["task_description"]
        old_exp = TASK_MEMORY[eq_root_id]["exp"]
        print("visited! reload task_name, task_description and experience")

    # < skip learning or start learning
    if eq_src_id != -1 and TASK_MEMORY[eq_root_id]["suc_num"] >= 3:  # visited
        # < skip
        final_exp = old_exp
        print("suc_num = %d, skip learning!" % TASK_MEMORY[eq_root_id]["suc_num"])
        print("final_exp = old_exp")
        # /> skip
    else:
        try:
            assert eq_src_id != -1
            tmp_i = TASK_MEMORY[eq_root_id]["suc_num"]
        except:
            tmp_i = 0
        print("suc_num = %d, continue learning!" % tmp_i)
        # < start learning

        # < exp transfer
        if TASK_MEMORY is None:
            print("memory is None, skip exp transfer")
            # < skip transfer
            transferred_exp = None
            # /> skip transfer
        else:
            # < exp transfer from memory
            print("memory is not None, start exp transfer")

            # top_indices is reused here,remove eq task
            if eq_src_id != -1:
                top_indices.pop(eq_src_id - 1)

            # < select src task
            if len(top_indices) == 0:
                selected_src_ids = []
            else:
                tmp_info = "<Target Task>\n%s\n</Target Task>" % task_description
                for order_id, selected_ind in enumerate(top_indices):
                    tmp_info += "\n\n<Candidate Task %d>\n%s\n</Candidate Task %d>" \
                                % (order_id + 1,
                                   TASK_MEMORY[selected_ind]["task_description"],
                                   order_id + 1)
                is_src_task_prompt = """%s

You are an outstanding source task retriever, capable of discovering source tasks related to the target task from the above candidate tasks.
The experience gained from solving the source tasks should be transferable to the target task.
Use the following JSON format to output the selected source tasks:
```json
{
"selected task ids": [ /* ids of selected source tasks. If there are no suitable source tasks, please return an empty list. */ ]
}
```
""" % tmp_info
                # gpt-3.5-turbo-1106 gpt-4-1106-preview
                # for model_type in ("gpt-3.5-turbo-1106","gpt-4-1106-preview"):
                max_retry = 5
                for i in range(max_retry + 1):
                    if i == max_retry:
                        raise Warning("retry too many times")
                    try:
                        # model_type = args.model_type
                        model_type = args.model_type
                        response = chat_create(model=model_type, max_tokens=128, seed=1 + i, temperature=1,
                                               messages=[{"role": "user", "content": is_src_task_prompt}], log_flag="is_src_task_prompt %d" % i)
                        res = re.findall("```json(.*?)```", response, flags=re.S)
                        if len(res) == 0:
                            res = re.findall(r"(\{.*?})", response, flags=re.S)
                        tmp_out = json.loads(res[0])
                        assert "selected task ids" in tmp_out
                        selected_src_ids = tmp_out["selected task ids"]
                        assert type(selected_src_ids) == list
                        selected_src_ids = [int(t) for t in selected_src_ids]
                        break
                    except Exception as e:
                        print("retry %d/%d", i + 1, max_retry)
                        traceback.print_exc()
            print("select src task = %s" % str(selected_src_ids))
            # /> select src task
            # < multi src tranfer
            if len(selected_src_ids) == 0:
                if eq_src_id != -1:
                    transferred_exp = old_exp
                    print("no src task, has old_exp, use old_exp as transferred_exp")
                else:
                    transferred_exp = None
                    print("no src task, no old_exp, use None as transferred_exp")

            else:
                print("has src task, start multi src transfer")
                # multi src exp transfer
                tmp_info = "<Target Task>\n%s\n</Target Task>" % task_description
                for order_id, selected_ind in enumerate(selected_src_ids):
                    selected_ind = top_indices[selected_ind - 1]
                    tmp_info += "\n\n<Source Task %d>\nTask Description: %s\nTask Experience:\n%s\n</Source Task %d>" \
                                % (order_id + 1,
                                   TASK_MEMORY[selected_ind]["task_description"],
                                   TASK_MEMORY[selected_ind]["exp"],
                                   order_id + 1)
                multi_src_transfer_prompt = """You are an excellent experience transfer agent, adept at transferring experience from one or more source tasks to the target task.
Here is the task description of the target task, as well as the task description and task experience of source tasks.

%s

Please follow the steps below to transfer experience:

Step 1: Task Understanding
Thoroughly understand the target task and source tasks, clearly identifying the commonalities and differences between them.

Step 2: Identify General Experience
Extracting general experience from the source tasks that can also be applied to the target task, especially insights that are common across multiple source tasks.
Avoid using task-specific experience from the source tasks that may not be relevant to the target task.
Be cautious of experience effective in the source tasks but could lead to errors in the target task.
Pay attention to the differences between the source and target tasks.

Step 3: Experience Adaptation
Adapt the general experience identified in Step 2 to the target task, adjusting for aspects that do not align perfectly with the target task's conditions and meeting the specific requirements of the target task.
Ensure that the experience provided are CLEAR, DETAILED, and GENERALLY APPLICABLE to unseen examples in the target task.
Use the following JSON format to output the adapted experience:
```json
{
"How to better accomplish the task or avoid low-quality responses": [ no more than 20 insights ],
"The specific process for handling this task": [ no more than 20 insights ]
}
```

Let's think step by step.
""" % tmp_info
                # gpt-3.5-turbo-1106 gpt-4-1106-preview
                # for model_type in ("gpt-3.5-turbo-1106","gpt-4-1106-preview"):
                max_retry = 100
                for i in range(max_retry + 1):
                    if i == max_retry:
                        raise Warning("retry too many times")
                    try:
                        # model_type = args.model_type
                        model_type = args.model_type
                        response = chat_create(model=model_type, max_tokens=2048, seed=1 + i, temperature=1,
                                               messages=[{"role": "user", "content": multi_src_transfer_prompt}
                                                         ], log_flag="multi_src_transfer_prompt %d" % i)
                        res = re.findall("```json(.*?)```", response, flags=re.S)
                        if len(res) == 0:
                            res = re.findall(r"(\{.*?})", response, flags=re.S)
                        trans_exp = None
                        for t in res:
                            try:
                                tmp_out = json.loads(t)

                                trans_exp = {
                                    "How to better accomplish the task or avoid low-quality responses": tmp_out[
                                        "How to better accomplish the task or avoid low-quality responses"],
                                    "The specific process for handling this task": tmp_out["The specific process for handling this task"]
                                }
                                break
                            except:
                                continue

                        if trans_exp is None:
                            tmptmptmpflag = 0
                            trans_exp = {
                                "How to better accomplish the task or avoid low-quality responses": [],
                                "The specific process for handling this task": []
                            }
                            for tmptmptmpline in response.strip().split("\n"):
                                tmptmptmpline = tmptmptmpline.strip()
                                if "How to better accomplish the task or avoid low-quality responses" in tmptmptmpline:
                                    assert tmptmptmpflag == 0
                                    tmptmptmpflag = 1
                                elif tmptmptmpflag == 1:
                                    if tmptmptmpline in ("[", "{"):
                                        continue
                                    elif re.fullmatch("[^a-zA-Z0-9]*", tmptmptmpline):
                                        tmptmptmpflag = 0
                                    else:
                                        assert re.match("([0-9]+\.?)|\- *", tmptmptmpline)
                                        trans_exp["How to better accomplish the task or avoid low-quality responses"].append(tmptmptmpline)
                                elif "The specific process for handling this task" in tmptmptmpline:
                                    assert tmptmptmpflag == 0
                                    tmptmptmpflag = 2
                                elif tmptmptmpflag == 2:
                                    if tmptmptmpline in ("[", "{"):
                                        continue
                                    elif re.fullmatch("[^a-zA-Z0-9]*", tmptmptmpline):
                                        tmptmptmpflag = 0
                                    else:
                                        assert re.match("([0-9]+\.?)|\- *", tmptmptmpline)
                                        trans_exp["The specific process for handling this task"].append(tmptmptmpline)

                        assert trans_exp is not None
                        assert len(trans_exp["The specific process for handling this task"]) != 0
                        assert len(enc.encode(response[:response.rfind("How to better accomplish the task")])) > 100
                        trans_exp = json.dumps(trans_exp, indent=2, ensure_ascii=False).strip()
                        break
                    except Exception as e:
                        print("retry %d/%d", i + 1, max_retry)
                        traceback.print_exc()
                # end with get trans_exp

                if eq_src_id == -1:
                    transferred_exp = trans_exp
                    print("get tran_exp, no old_exp, just use tran_exp as transferred_exp")
                else:
                    print("get tran_exp, has old_exp, start merge")
                    # merge trans_exp and old_exp

                    unmerge_exp1 = json.loads(old_exp)
                    unmerge_exp2 = json.loads(trans_exp)
                    old_trans_merge_prompt = """<Target Task>
%s
</Target Task>

<Existing Experience>
%s
</Existing Experience>

You are an excellent experience refiner. Please help me refine the above existing experiences related to the target task.
1. For "How to better accomplish the task or avoid low-quality responses", please integrate insights by combining those that are closely related and eliminating any repetitions
2. Please integrate the above "Task Processing Flow 1" and "Task Processing Flow 2" into one unified workflow process. Ensure that the primary goals and functionality of both original processes are preserved; Effectively resolve possible conflicts or overlaps between the two processes.
Use the following JSON format to output refined target task experience:
```json
{
"How to better accomplish the task or avoid low-quality responses": [ no more than 20 insights ],
"The specific process for handling this task": [ no more than 20 insights ]
}
```
""" % (task_description, json.dumps({
                        "How to better accomplish the task or avoid low-quality responses": unmerge_exp1[
                                                                                                "How to better accomplish the task or avoid low-quality responses"] +
                                                                                            unmerge_exp2[
                                                                                                "How to better accomplish the task or avoid low-quality responses"],
                        "Task Processing Flow 1": unmerge_exp1["The specific process for handling this task"],
                        "Task Processing Flow 2": unmerge_exp2["The specific process for handling this task"]
                    }, ensure_ascii=False, indent=2).strip())
                    # gpt-3.5-turbo-1106 gpt-4-1106-preview
                    # for model_type in ("gpt-3.5-turbo-1106","gpt-4-1106-preview"):
                    max_retry = 5
                    for i in range(max_retry + 1):
                        if i == max_retry:
                            raise Warning("retry too many times")
                        try:
                            # model_type = args.model_type
                            model_type = args.model_type
                            response = chat_create(model=model_type, max_tokens=2048, seed=1 + i, temperature=1,
                                                   messages=[{"role": "user", "content": old_trans_merge_prompt}], log_flag="old_trans_merge_prompt %d" % i)
                            res = re.findall("```json(.*?)```", response, flags=re.S)
                            if len(res) == 0:
                                res = re.findall(r"(\{.*?})", response, flags=re.S)
                            transferred_exp = None
                            for t in res:
                                try:
                                    tmp_out = json.loads(t)

                                    transferred_exp = {
                                        "How to better accomplish the task or avoid low-quality responses": tmp_out[
                                            "How to better accomplish the task or avoid low-quality responses"],
                                        "The specific process for handling this task": tmp_out["The specific process for handling this task"]
                                    }
                                    break
                                except:
                                    continue
                            assert transferred_exp is not None
                            transferred_exp = json.dumps(transferred_exp, indent=2, ensure_ascii=False).strip()
                            break
                        except Exception as e:
                            print("retry %d/%d", i + 1, max_retry)
                            traceback.print_exc()
                    print("end merge, use merged_exp as transferred_exp")
                # end with get transferred_exp
            # /> exp transfer from memory
        print("end full exp transfer process")
        # /> exp transfer

        # start auto practice
        # 1. auto question generate
        valid_paras = item["web_texts"]
        print("start auto practice process, num of valid_paras = %d" % len(valid_paras))
        auto_instance_list = []
        for refid, ref_text in enumerate(valid_paras):
            ref_text = ref_text.strip()
            auto_question_prompt = r"""<Reference Text>
%s
</Reference Text>

<Example Question>
%s
</Example Question>

<Task Type of the Example Question>
%s
</Task Type of the Example Question>

You are an excellent questioner.
Please carefully read the reference text provided above and formulate a new question based on it.
The new question must maintain the same expression style, structure, and required output format as the example question.
The new question must belong to the same task type of the example question.
The new question must be well-defined, with a complete and clear description that can be answered and at least one correct answer exists.
You are forbidden from providing answers to your new question.
Use the following format to output your answer:
<New Question>
/* Your new question. */
</New Question>

""" % (ref_text, prompt, task_description)
            # gpt-3.5-turbo-1106 gpt-4-1106-preview
            # for model_type in ("gpt-3.5-turbo-1106","gpt-4-1106-preview"):
            max_retry = 20
            for i in range(max_retry + 1):
                if i == max_retry:
                    raise Warning("retry too many times")
                try:
                    #
                    model_type = args.model_type
                    response = chat_create(model=model_type, max_tokens=1024, seed=1 + i, temperature=1,
                                           messages=[{"role": "user", "content": auto_question_prompt}], log_flag="auto_question_prompt %d" % i)
                    res = re.findall(r"""<New Question>(.*?)</New Question>""", response, flags=re.S)
                    new_q = res[0].strip()
                    break
                except Exception as e:
                    print("retry %d/%d", i + 1, max_retry)
                    traceback.print_exc()
            # get new_q

            # 2. auto practice new question
            auto_practice_prompt = ""
            if transferred_exp is not None:
                auto_practice_prompt += """<Task Experience>
%s
</Task Experience>
Please refer to the above experience to answer the following question.

""" % transferred_exp
            auto_practice_prompt += new_q.strip() + "\n\nPlease provide specific, detailed, and comprehensive steps of your thought."
            # gpt-3.5-turbo-1106 gpt-4-1106-preview
            # for model_type in ("gpt-3.5-turbo-1106","gpt-4-1106-preview"):
            max_retry = 20
            for i in range(max_retry + 1):
                if i == max_retry:
                    raise Warning("retry too many times")
                try:
                    #
                    model_type = args.model_type
                    response = chat_create(model=model_type, max_tokens=1024, seed=1 + i, temperature=1,
                                           messages=[{"role": "user", "content": auto_practice_prompt}],
                                           log_flag="auto_practice_prompt %d" % i)
                    break
                except Exception as e:
                    print("retry %d/%d", i + 1, max_retry)
                    traceback.print_exc()
            cur_response = response.strip()
            # get cur_response

            # 3. auto practice verification

            auto_verification_prompt = """<Reference Text>
%s
</Reference Text>

<Target Question>
%s
</Target Question>

<Reasoning Process and Answer>
%s
</Reasoning Process and Answer>
        
You are an outstanding checker, skilled at examining the reasoning process and the correctness of the answer of the target question based on the reference text.
Pay close attention to whether the reasoning process and the answer are consistent or inconsistent with the reference text.
Use the following JSON format to output your opinion:
```json
{
"correctness": /* "correct", "wrong" or "inconclusive" */
}
```

Let's think step by step.
""" % (ref_text, new_q, cur_response)
            # gpt-3.5-turbo-1106 gpt-4-1106-preview
            # for model_type in ("gpt-3.5-turbo-1106","gpt-4-1106-preview"):

            vote_log = []
            for vote_id in range(4):
                if len(vote_log) != len(set(vote_log)):
                    break
                max_retry = 20
                for i in range(max_retry + 1):
                    if i == max_retry:
                        raise Warning("retry too many times")
                    try:
                        #
                        model_type = args.model_type
                        response = chat_create(model=model_type, max_tokens=512, seed=1 + i + 1000 * vote_id, temperature=1,
                                               messages=[{"role": "user", "content": auto_verification_prompt}], log_flag="auto_verification_prompt %d" % i)
                        res = re.findall("```json(.*?)```", response, flags=re.S)
                        if len(res) == 0:
                            res = re.findall(r"(\{.*?})", response, flags=re.S)
                        veri_res = None
                        for t in res:
                            try:
                                tmp_out = json.loads(t)
                                cur_correctness = tmp_out["correctness"]
                                veri_res = ("wrong", "correct", "inconclusive").index(cur_correctness.lower())
                                break
                            except:
                                continue

                        if veri_res is None:
                            rule_based_p_text = re.sub("\t| ", "", response)
                            # "correctness": "correct"
                            if "\"correctness\":\"wrong\"" in rule_based_p_text or "\"correctness\":\"incorrect\"" in rule_based_p_text:
                                veri_res = 0
                            elif "\"correctness\":\"correct\"" in rule_based_p_text:
                                veri_res = 1
                            elif "\"correctness\":\"inconclusive\"" in rule_based_p_text:
                                veri_res = 2
                            else:
                                rule_based_p_text = response.strip().split("\n")
                                rule_based_p_text = [t for t in rule_based_p_text if t.strip() != ""][-1]
                                if "\"wrong\"" in rule_based_p_text or "\"wrong.\"" in rule_based_p_text or "the answer is considered wrong" in rule_based_p_text or "the correctness is wrong" in rule_based_p_text:
                                    veri_res = 0
                                elif "\"correct\"" in rule_based_p_text or "\"correct.\"" in rule_based_p_text or "the answer is considered correct" in rule_based_p_text or "the correctness is correct" in rule_based_p_text:
                                    veri_res = 1
                                elif "\"inconclusive\"" in rule_based_p_text or "\"inconclusive.\"" in rule_based_p_text or "the answer is considered inconclusive" in rule_based_p_text or "the correctness is inconclusive" in rule_based_p_text:
                                    veri_res = 2
                                elif "\"incorrect\"" in rule_based_p_text or "\"incorrect.\"" in rule_based_p_text or "the answer is considered incorrect" in rule_based_p_text or "the correctness is incorrect" in rule_based_p_text:
                                    veri_res = 0
                        assert veri_res is not None
                        vote_log.append(veri_res)
                        break
                    except Exception as e:
                        print("retry %d/%d", i + 1, max_retry)
                        traceback.print_exc()
            assert len(vote_log) != len(set(vote_log))
            veri_res = find_count_max(vote_log)
            if veri_res == 2:
                print("vote result = inconclusive, skip this instance")
                continue
            auto_instance_list.append((new_q, cur_response, veri_res))
            print("vote result = %s, add this instance" % ("wrong", "correct", "inconclusive")[veri_res])

            if len(auto_instance_list) >= 5: break  # auto practice 5 times per query
        # end with get auto_instance_list
        print("end auto practice process, num of pos:neg:all = %d : %d : %d" % (
            sum([t[2] == 1 for t in auto_instance_list]), sum([t[2] == 0 for t in auto_instance_list]), len(auto_instance_list)))

        # start exp induction
        print("start exp induction process")
        correct_examples, incorrect_examples = [], []
        for new_q, cur_response, veri_res in auto_instance_list:
            if veri_res == 1:
                order_id = len(correct_examples)
                correct_examples.append(
                    "<Correct Example %d>\n<Question>\n%s\n</Question>\n<Reasoning Process and Answer>\n%s\n</Reasoning Process and Answer>\n</Correct Example %d>" \
                    % (order_id + 1,
                       new_q,
                       cur_response,
                       order_id + 1))
            else:
                order_id = len(incorrect_examples)
                incorrect_examples.append(
                    "<Incorrect Example %d>\n<Question>\n%s\n</Question>\n<Reasoning Process and Answer>\n%s\n</Reasoning Process and Answer>\n</Incorrect Example %d>" \
                    % (order_id + 1,
                       new_q,
                       cur_response,
                       order_id + 1))
        auto_example_history = "\n\n".join(correct_examples + incorrect_examples)

        exp_induction_prompt = """You are an excellent experiential summarizer, adept at extracting task-solving insights from examples of the target task.
Here are several target task examples with correct or incorrect answers:
%s

Based on the examples provided above, please follow the steps below to summarize the experience:

Step1: Observe and Analyze the Examples
Summarize the commonalities in the correct examples, identify patterns in the incorrect examples, and compare the differences between the correct and incorrect examples.

Step2: Summarize Experience
Based on the observations and analysis from the Step1, summarize task-solving insights.
Ensure that the insights provided are CLEAR, DETAILED, and are GENERALLY APPLICABLE to unseen examples of the target task. 
Use the following JSON format to output the summarized experience:
```json
{
"How to better accomplish the task or avoid low-quality responses": [ no more than 20 insights ],
"The specific process for handling this task": [ no more than 20 insights ]
}
```

Let's think step by step.
""" % (auto_example_history)

        # gpt-3.5-turbo-1106 gpt-4-1106-preview
        # for model_type in ("gpt-3.5-turbo-1106","gpt-4-1106-preview"):
        max_retry = 20
        for i in range(max_retry + 1):
            if i == max_retry:
                raise Warning("retry too many times")
            try:
                #
                model_type = args.model_type
                response = chat_create(model=model_type, max_tokens=2048, seed=1 + i, temperature=1,
                                       messages=[{"role": "user", "content": exp_induction_prompt}
                                                 ], log_flag="exp_induction_prompt %d" % i)
                res = re.findall("```json(.*?)```", response, flags=re.S)
                if len(res) == 0:
                    res = re.findall(r"(\{.*?})", response, flags=re.S)
                induced_exp = None
                for t in res:
                    try:
                        tmp_out = json.loads(t)
                        induced_exp = {
                            "How to better accomplish the task or avoid low-quality responses": tmp_out[
                                "How to better accomplish the task or avoid low-quality responses"],
                            "The specific process for handling this task": tmp_out["The specific process for handling this task"]
                        }
                        break
                    except:
                        continue

                if induced_exp is None:
                    tmptmptmpflag = 0
                    induced_exp = {
                        "How to better accomplish the task or avoid low-quality responses": [],
                        "The specific process for handling this task": []
                    }
                    for tmptmptmpline in response.strip().split("\n"):
                        tmptmptmpline = tmptmptmpline.strip()
                        if "How to better accomplish the task or avoid low-quality responses" in tmptmptmpline:
                            assert tmptmptmpflag == 0
                            tmptmptmpflag = 1
                        elif tmptmptmpflag == 1:
                            if tmptmptmpline in ("[", "{"):
                                continue
                            elif re.fullmatch("[^a-zA-Z0-9]*", tmptmptmpline):
                                tmptmptmpflag = 0
                            else:
                                assert re.match("([0-9]+\.?)|\- *", tmptmptmpline)
                                induced_exp["How to better accomplish the task or avoid low-quality responses"].append(tmptmptmpline)
                        elif "The specific process for handling this task" in tmptmptmpline:
                            assert tmptmptmpflag == 0
                            tmptmptmpflag = 2
                        elif tmptmptmpflag == 2:
                            if tmptmptmpline in ("[", "{"):
                                continue
                            elif re.fullmatch("[^a-zA-Z0-9]*", tmptmptmpline):
                                tmptmptmpflag = 0
                            else:
                                assert re.match("([0-9]+\.?)|\- *", tmptmptmpline)
                                induced_exp["The specific process for handling this task"].append(tmptmptmpline)
                assert induced_exp is not None
                assert len(induced_exp["The specific process for handling this task"]) != 0
                assert len(enc.encode(response[:response.rfind("How to better accomplish the task")])) > 30
                induced_exp = json.dumps(induced_exp, indent=2, ensure_ascii=False).strip()
                break
            except Exception as e:
                print("retry %d/%d", i + 1, max_retry)
                traceback.print_exc()
        # /> finish exp induction
        print("end exp induction process")

        if transferred_exp is None:
            print("no transferred_exp, final_exp=induced_exp")
            final_exp = induced_exp
        else:
            print("has transferred_exp, start merge, final_exp=transferred_induced_exp")

            # merge transferred_exp, induced_exp

            unmerge_exp1 = json.loads(transferred_exp)
            unmerge_exp2 = json.loads(induced_exp)
            old_trans_merge_prompt = """<Target Task>
%s
</Target Task>

<Existing Experience>
%s
</Existing Experience>

You are an excellent experience refiner. Please help me refine the above existing experiences related to the target task.
1. For "How to better accomplish the task or avoid low-quality responses", please integrate insights by combining those that are closely related and eliminating any repetitions
2. Please integrate the above "Task Processing Flow 1" and "Task Processing Flow 2" into one unified workflow process. Ensure that the primary goals and functionality of both original processes are preserved; Effectively resolve possible conflicts or overlaps between the two processes.
Use the following JSON format to output refined target task experience:
```json
{
"How to better accomplish the task or avoid low-quality responses": [ no more than 20 insights ],
"The specific process for handling this task": [ no more than 20 insights ]
}
```
""" % (task_description, json.dumps({
                "How to better accomplish the task or avoid low-quality responses": unmerge_exp1[
                                                                                        "How to better accomplish the task or avoid low-quality responses"] +
                                                                                    unmerge_exp2[
                                                                                        "How to better accomplish the task or avoid low-quality responses"],
                "Task Processing Flow 1": unmerge_exp1["The specific process for handling this task"],
                "Task Processing Flow 2": unmerge_exp2["The specific process for handling this task"]
            }, ensure_ascii=False, indent=2).strip())

            # gpt-3.5-turbo-1106 gpt-4-1106-preview
            # for model_type in ("gpt-3.5-turbo-1106","gpt-4-1106-preview"):
            max_retry = 5
            for i in range(max_retry + 1):
                if i == max_retry:
                    raise Warning("retry too many times")
                try:
                    #
                    model_type = args.model_type
                    response = chat_create(model=model_type, max_tokens=2048, seed=1 + i, temperature=1,
                                           messages=[{"role": "user", "content": old_trans_merge_prompt}], log_flag="old_trans_merge_prompt %d" % i)
                    res = re.findall("```json(.*?)```", response, flags=re.S)
                    if len(res) == 0:
                        res = re.findall(r"(\{.*?})", response, flags=re.S)
                    transferred_induced_exp = None
                    for t in res:
                        try:
                            tmp_out = json.loads(t)

                            transferred_induced_exp = {
                                "How to better accomplish the task or avoid low-quality responses": tmp_out[
                                    "How to better accomplish the task or avoid low-quality responses"],
                                "The specific process for handling this task": tmp_out["The specific process for handling this task"]
                            }
                            break
                        except:
                            continue
                    assert transferred_induced_exp is not None
                    transferred_induced_exp = json.dumps(transferred_induced_exp, indent=2, ensure_ascii=False).strip()
                    break
                except Exception as e:
                    print("retry %d/%d", i + 1, max_retry)
                    traceback.print_exc()
            print("end merge")
            final_exp = transferred_induced_exp
        # update memory

        suc_flag = int(len(incorrect_examples) == 0)

        tmptmp_exp_unpack = json.loads(final_exp)
        cur_exp1_num = len(tmptmp_exp_unpack["How to better accomplish the task or avoid low-quality responses"])
        cur_exp2_num = len(tmptmp_exp_unpack["The specific process for handling this task"])
        cur_exp_all_num = cur_exp1_num + cur_exp2_num

        print("update memory, is_suc = %s" % str(suc_flag))
        if TASK_MEMORY is None:
            print("memory is None, init memory")
            task_description_embeds = \
                ctx_encoder(**ctx_tokenizer(task_description, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device))[
                    "pooler_output"][0].cpu().numpy()
            TASK_MEMORY = Dataset.from_dict({"task_name": [task_name], "task_description": [task_description], "prompt": [prompt],
                                             "embeddings": [task_description_embeds], "exp": [final_exp], "suc_num": [suc_flag],
                                             "learned_num": [1],
                                             "exp_1_num": [cur_exp1_num], "exp_2_num": [cur_exp2_num], "exp_all_num": [cur_exp_all_num]
                                             })
        elif eq_src_id == -1:
            print("memory is not None, unvisited, add new task into memory")
            task_description_embeds = \
                ctx_encoder(**ctx_tokenizer(task_description, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device))[
                    "pooler_output"][0].cpu().tolist()
            TASK_MEMORY = TASK_MEMORY.to_dict()
            TASK_MEMORY["task_name"].append(task_name)
            TASK_MEMORY["task_description"].append(task_description)
            TASK_MEMORY["prompt"].append(prompt)
            TASK_MEMORY["embeddings"].append(task_description_embeds)
            TASK_MEMORY["exp"].append(final_exp)
            TASK_MEMORY["suc_num"].append(suc_flag)
            TASK_MEMORY["learned_num"].append(1)
            TASK_MEMORY["exp_1_num"].append(cur_exp1_num)
            TASK_MEMORY["exp_2_num"].append(cur_exp2_num)
            TASK_MEMORY["exp_all_num"].append(cur_exp_all_num)
            TASK_MEMORY = Dataset.from_dict(TASK_MEMORY)
            TASK_MEMORY.add_faiss_index(column='embeddings')
        else:
            print("memory is not None, visited, update task in memory, eq_root_id = %d" % eq_root_id)
            TASK_MEMORY = TASK_MEMORY.to_dict()
            TASK_MEMORY["exp"][eq_root_id] = final_exp
            if suc_flag == 1:
                TASK_MEMORY["suc_num"][eq_root_id] += suc_flag
            else:
                TASK_MEMORY["suc_num"][eq_root_id] = 0
            TASK_MEMORY["learned_num"][eq_root_id] += 1

            TASK_MEMORY["exp_1_num"][eq_root_id] = cur_exp1_num
            TASK_MEMORY["exp_2_num"][eq_root_id] = cur_exp2_num
            TASK_MEMORY["exp_all_num"][eq_root_id] = cur_exp_all_num
            TASK_MEMORY = Dataset.from_dict(TASK_MEMORY)
            TASK_MEMORY.add_faiss_index(column='embeddings')

        print("end update memory")

        # /> finish learning

    # /> skip learning or finish learning
    assert final_exp is not None
    item["final_exp"] = final_exp
    exp_data_recorder.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("######################### END %d #########################" % item_id)
    ...

    # end MAIN
exp_data_recorder.close()

pdb.set_trace()



