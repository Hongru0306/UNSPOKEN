import os, re
import json
from openai import OpenAI
import numpy as np

def extract_answer(sentence):
    pattern = r'[A-F]'
    matches = re.findall(pattern, sentence)
    
    # 去重并保留顺序
    seen = set()
    unique = []
    for char in matches:
        if char not in seen:
            seen.add(char)
            unique.append(char)
    
    # 构造递增序列
    result = []
    for char in unique:
        if not result or char > result[-1]:
            result.append(char)
    
    return result

def extract_answer_llm(sentence):
    prompt = """
You are a professional answer extraction expert. Analyze the given text to identify the CORRECT ANSWER selected by the model. 
Extract ONLY the final chosen answer option letter (A/B/C/D/E/F) from the analysis.

if the model chooses the answer "A", you should return ["A"] in the JSON format.
if the model chooses the answer "A/B" or "AB", you should return ["A", "B"] in the JSON format.

Text to analyze:

[ sentence ]

Instructions:
1. Identify which option is explicitly stated as the correct answer
2. Ignore all intermediate reasoning and analysis
3. Return ONLY the letter inside a JSON list
4. Ensure case sensitivity (use uppercase letters)

Return valid JSON format with key "answer":
"""
    client = OpenAI(
        api_key='', 
        base_url="",
    )
    completion = client.chat.completions.create(
        model="qwen-max-2025-01-25",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt.replace('[ sentence ]', sentence)},
            ],
        )
    response = completion.choices[0].message.content
    response = response.replace('```json', '').replace('```', '')
    try:
        response = json.loads(response)
        return response['answer']
    except json.JSONDecodeError:
        return None

def calculate_multi_score_iter(data):
    total_score = 0.0
    for item in data:
        if not item['model_ans']:
            continue
        model = set(item['model_ans'])
        correct = set(item['correct_ans'])

        if not model.issubset(correct):
            continue  # 错选，得0分
        else:
            # 计算正确选中的比例
            correct_count = len(model)
            total_correct = len(correct)
            if total_correct == 0:
                # 避免除以0，假设正确答案至少有一个选项
                score = 0.0
            else:
                score = correct_count / total_correct
            total_score += score
    # 计算平均分并转换为百分比
    return (total_score / len(data)) * 100 if data else 0.0

def calculate_multi_score(model_ans, correct_ans):
    '''
    model_ans: list of str
    correct_ans: list of str
    '''
    if not model_ans:
        return 0
    model = set(model_ans)
    correct = set(correct_ans)

    if not model.issubset(correct):
        return 0  # 错选，得0分
    else:
        # 计算正确选中的比例
        correct_count = len(model)
        total_correct = len(correct)
        if total_correct == 0:
            # 避免除以0，假设正确答案至少有一个选项
            return 0
        else:
            score = correct_count / total_correct
            return round(score, 2)

def calculate_single_score_iter(data):
    total_score = 0.0
    for item in data:
        if item['model_ans'] == item['correct_ans']:
            total_score += 1
    return (total_score / len(data)) * 100

def calculate_single_score(model_ans, correct_ans):
    if model_ans == correct_ans:
        return 1
    return 0


def eval_score(path):

    json_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        task = data['task']
        model_name = data['model_name']
        single_result = data['single_res']
        multi_result = data['multi_res']

        single = {
            'zh': {
                'score': [],
            },
            'en': {
                'score': [],
            },
        }

        multi = {
            'zh': {
                'score': [],
            },
            'en': {
                'score': [],
            },
        }

        for item in single_result:
            label = item["label"]
            langeue = item['language']

            if single[langeue].get(label) is None:
                single[langeue][label] = [item['score']]
            else:
                single[langeue][label].append(item['score'])

            single[langeue]['score'].append(item['score'])

        for item in multi_result:
            label = item["label"]
            langeue = item['language']

            if multi[langeue].get(label) is None:
                multi[langeue][label] = [item['score']]
            else:
                multi[langeue][label].append(item['score'])

            multi[langeue]['score'].append(item['score'])
        
        labels = list(single['zh'].keys())

        # 中文各类别平均分
        zh_single_score = {}
        en_single_score = {}
        for label in labels:
            zh_single_score[label] = round(np.mean(single['zh'][label]), 3)
            en_single_score[label] = round(np.mean(single['en'][label]), 3)
        

        zh_multi_score = {}
        en_multi_score = {}
        for label in labels:
            zh_multi_score[label] = round(np.mean(multi['zh'][label]), 3)
            en_multi_score[label] = round(np.mean(multi['en'][label]), 3)

        # print(f'########################### {model_name} {task} ###########################\n')
        # print(f"Chinese Single : {zh_single_score['score']}, English Single : {en_single_score['score']}")
        # print(f"Chinese Muiti : {zh_multi_score['score']}, Chinese Muiti : {en_multi_score['score']}")
        # print("\n---------------------- Chinese Single & Mutli ----------------------")
        # for label in labels:
        #     print(f"{label} : single - {zh_single_score[label]} | multi - {zh_multi_score[label]}")

        # print("\n---------------------- English Single & Mutli ----------------------")
        # for label in labels:
        #     print(f"{label} : single - {en_single_score[label]} | multi - {en_multi_score[label]}")

        # print(f'################################### End ###################################\n')


        # 顶部装饰线
        print(f'\n\033[1;36m{"=" * 30} {model_name} {task} {"=" * 30}\033[0m')  # 青色加粗标题

        # 总分展示
        print(f"\n\033[1;33m• Overall Scores:\033[0m")  # 黄色标题
        print(f"\033[94mChinese\033[0m  Single: \033[1m{zh_single_score['score']:.3f}\033[0m"
            f"  |  Multi: \033[1m{zh_multi_score['score']:.3f}\033[0m")
        print(f"\033[94mEnglish\033[0m  Single: \033[1m{en_single_score['score']:.3f}\033[0m"
            f"  |  Multi: \033[1m{en_multi_score['score']:.3f}\033[0m")

        # 中文详细分数
        print(f"\n\033[1;32m〔 Chinese Details 〕\033[0m")  # 绿色标题
        print(f"\033[90m{'Label':<10} | {'Single':<8} | {'Multi':<8}\033[0m")  # 灰色表头
        print(f"\033[90m{'-'*10}-|---------|---------\033[0m")
        for label in labels:
            print(f"{label:<10} | \033[93m{zh_single_score[label]:<8.3f}\033[0m | \033[93m{zh_multi_score[label]:<8.3f}\033[0m")

        # 英文详细分数
        print(f"\n\033[1;35m〔 English Details 〕\033[0m")  # 紫色标题
        print(f"\033[90m{'Label':<10} | {'Single':<8} | {'Multi':<8}\033[0m")
        print(f"\033[90m{'-'*10}-|---------|---------\033[0m")
        for label in labels:
            print(f"{label:<10} | \033[93m{en_single_score[label]:<8.3f}\033[0m | \033[93m{en_multi_score[label]:<8.3f}\033[0m")

        print(f"\n\033[1;36m{'=' * 80}\033[0m\n")
