from model import *
from utils import extract_answer, extract_answer_llm, calculate_multi_score_iter, calculate_multi_score, calculate_single_score_iter, calculate_single_score
from prompt import *
import json
import os
import pandas as pd
from tqdm import tqdm
import traceback
# import random

config = {
    'direct': {
        'zh': [DIRECT_SINGLE_ZH, DIRECT_MULTIPLE_ZH],
        'en': [DIRECT_SINGLE_EN, DIRECT_MULTIPLE_EN],
    },
    'cot': {
        'zh': [COT_SINGLE_ZH, COT_MULTIPLE_ZH],
        'en': [COT_SINGLE_EN, COT_MULTIPLE_EN],
    },
    'xlt': {
        'zh': [XLT_SINGLE_ZH, XLT_MULTIPLE_ZH],
        'en': [XLT_SINGLE_EN, XLT_MULTIPLE_EN],
    }
}

def run_experiment(model, df, task, path):
    """
    model: Model object
    df: DataFrame
    task: str
    path: str, path to audio files
    """
    if task == 'direct':
        prompts = config['direct']
    elif task == 'cot':
        prompts = config['cot']
    elif task == 'xlt':
        prompts = config['xlt']
    else:
        raise ValueError('Invalid task')
    
    if not os.path.exists('answer_output'):
        os.mkdir('answer_output')
    if not os.path.exists(f'answer_output/{task}'):
        os.mkdir(f'answer_output/{task}')
    
    single_res = []
    multi_res = []

    breakdowns_audio = []



    for i in tqdm(range(0, len(df))): # range(0, len(df))): # len(df))
        item = df.iloc[i]
        audio_path = path + '/' + item['save_path']
        # print(audio_path)
        
        single_question = item['single_question']
        single_select = item['single_choice']
        single_select_ans = item['single_answer']
        single_select_ans = extract_answer(single_select_ans)[0]

        multi_question = item['multiple_question']
        multi_select = item['multi_choice']
        multi_select_ans = item['multi_answer']
        multi_select_ans = extract_answer(multi_select_ans)
        
        label = item['humor_label']

        if item['Language'] == 'zh':
            prompt = prompts['zh']
        else:
            prompt = prompts['en']

        try:
            model_single_response_raw = model.chat(prompt=prompt[0].format(question=single_question, choices=single_select), audio_path=audio_path)
            model_single_response = extract_answer_llm(model_single_response_raw)
            if not model_single_response:
                model_single_response = ''
            else:
                model_single_response = model_single_response[0]

            model_multi_response_raw = model.chat(prompt=prompt[1].format(question=multi_question, choices=multi_select), audio_path=audio_path)
            model_multi_response = extract_answer_llm(model_multi_response_raw)
            if not model_multi_response:
                model_multi_response = []
            
            single_score = calculate_single_score(model_single_response, single_select_ans)
            multi_score = calculate_multi_score(model_multi_response, multi_select_ans)
        except Exception as e:
            tqdm.write(f"\nError at Question {i+1}: {e}")
            traceback.print_exc()
            # **You can rerun the inferencing process for the broken audio files later to get better results**

            single_score = 0
            multi_score = 0
            model_single_response_raw = ''
            model_single_response = ''
            model_multi_response_raw = ''
            model_multi_response = ''
            # 将失败样本也记录到结果列表中（model_ans/model_ans_raw 置空，score 为 0）
            single_res.append({
                'save_path': audio_path,
                'model_ans_raw': model_single_response_raw,
                'model_ans': model_single_response,
                'correct_ans': single_select_ans,
                'label': label,
                'language': item['Language'],
                'score': single_score,
            })

            multi_res.append({
                'save_path': audio_path,
                'model_ans_raw': model_multi_response_raw,
                'model_ans': model_multi_response,
                'correct_ans': multi_select_ans,
                'label': label,
                'language': item['Language'],
                'score': multi_score,
            })

            breakdowns_audio.append(audio_path)
            print(f"\nQuestion {i+1} Audio Brokedown: {item['save_path']}\n")
            # 继续循环，保留失败样本在结果中以便后续统计
            continue

        single_res.append({
            'save_path': audio_path, 
            'model_ans_raw': model_single_response_raw,
            'model_ans': model_single_response, 
            'correct_ans': single_select_ans, 
            'label': label, 
            'language': item['Language'],
            'score': single_score,
        })

        multi_res.append({
            'save_path': audio_path, 
            'model_ans_raw': model_multi_response_raw,
            'model_ans': model_multi_response, 
            'correct_ans': multi_select_ans, 
            'label': label, 
            'language': item['Language'],
            'score': multi_score,
        })

        tqdm.write(f'################## Question {i + 1} ##################')
        tqdm.write(f"Langue: {item['Language']}")
        tqdm.write('               Single Select Question')
        tqdm.write(f'{model.model_name}: {model_single_response}')
        tqdm.write(f'Correct Answer: {single_select_ans}')
        tqdm.write(f'Score: {single_score}')

        tqdm.write('               Multi Select Question')
        tqdm.write(f'{model.model_name}: {model_multi_response}')
        tqdm.write(f'Correct Answer: {multi_select_ans}')
        tqdm.write(f'Score: {multi_score}')

    single_score = calculate_single_score_iter(single_res)
    multi_score = calculate_multi_score_iter(multi_res)

    result = {
        'task': task,
        'model_name': model.model_name,
        'single_score': single_score,
        'multi_score': multi_score,
        'single_res': single_res,
        'multi_res': multi_res,
    }

    with open(f'answer_output/{task}/{model.model_name}.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f'################## {task} Score ##################')
    print(f'{model.model_name} Single Select Score: {single_score:.2f}')
    print(f'{model.model_name} Multi Select Score: {multi_score:.2f}')

    print(f'Breakdowns Audio Number: {len(breakdowns_audio)}')

    with open(f'answer_output/{task}/{model.model_name}_breakdowns_audio.txt', 'w') as f:
        for item in breakdowns_audio:
            f.write(item + '\n')



if __name__ == '__main__':
    
    # model = GPT4oAudioPreview(api_key="", base_url="")
    model = Qwen2OmniAudio(model_path='')
    # model = Qwen2Audio(model_path='')
    # model = GeminiAudio(model_name='gemini-2.0-flash',api_key='')
    # model = GeminiText(model_name='gemini-1.5-pro',api_key='')
    df = pd.read_csv('./final_utf8.csv')
    path = './sliced_mp3'
    task = 'direct'  # direct, cot, xlt

    run_experiment(model, df, 'direct', path)