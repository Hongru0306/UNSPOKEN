from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from transformers.generation import GenerationConfig
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import torch
import base64
from zhipuai import ZhipuAI
from io import BytesIO
from urllib.request import urlopen
import librosa
import openai
import base64

import warnings
warnings.filterwarnings("ignore")

import soundfile as sf
try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    process_mm_info = None  # 若无 utils，可自定义音频处理



class GeminiAudio:
    """
    gemini-2.0-flash
    gemini-1.5-flash
    gemini-1.5-pro

    """

    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
        self.client = genai.Client(api_key=self.api_key)  

    def chat(self, prompt, audio_path):
        myfile = self.client.files.upload(file=audio_path)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                prompt,
                myfile,
            ]
        )

        return response.text


# Alterantive GeminiAudio Class using openai-compatible API

# class GeminiAudio:
#     """
#     gemini-2.0-flash
#     gemini-1.5-flash
#     gemini-1.5-pro

#     """

#     def __init__(self, model_name, api_key):
#         # model_name: e.g. "gpt-4o"
#         # api_key: API key for the openai-compatible endpoint
#         self.model_name = model_name
#         self.api_key = api_key
#         openai.api_key = self.api_key
#         openai.base_url = 'https://api.shubiaobiao.cn/v1/'  # 固定 base_url
#         self.base_url = openai.base_url

#     def chat(self, prompt, audio_path):
#         """
#         使用 openai 兼容接口发送带音频的请求。

#         - prompt: 文本提示（string）
#         - audio_path: 本地音频文件路径（mp3）

#         返回模型回复的文本（或原始 response，如果无法解析）。
#         """

#         # 读取并 base64 编码音频文件（与示例一致）
#         with open(audio_path, "rb") as audio_file:
#             mp3_data = audio_file.read()
#         encoded_string = base64.b64encode(mp3_data).decode('utf-8')

#         try:
#             response = openai.chat.completions.create(
#                 model=self.model_name,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": [
#                         {"type": "text", "text": prompt},
#                         {"type": "input_audio", "input_audio": {"data": encoded_string, "format": "mp3"}}
#                     ]}
#                 ]
#             )
#         except Exception as e:
#             # 将底层异常封装成 RuntimeError 以便上层统一处理
#             raise RuntimeError(f"OpenAI-compatible API 请求失败: {e}") from e

#         # 尝试根据不同实现解析常见字段
#         try:
#             # 一些实现返回字典风格
#             return response['choices'][0]['message']['content']
#         except Exception:
#             pass

#         try:
#             # 原生 openai-python 风格对象访问
#             return response.choices[0].message.audio.transcript
#         except Exception:
#             pass

#         # 无法解析，返回空字符串，避免后续replace报错
#         return str(response)




class GPT4oAudioPreview:
    model_name = 'gpt-4o-mini-audio-preview'  # gpt-4o-mini-audio-preview

    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        # self.mini = mini
        self.init_clinet()
        # if self.mini:
        #     self.model = 'gpt-4o-mini-audio-preview'
        # else:
        #     self.model = 'gpt-4o-audio-preview'
        # print(f"Using model: {self.model}")

    def init_clinet(self):
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
    def chat(self, prompt, audio_path):
        with open(audio_path, "rb") as audio_file:
            mp3_data = audio_file.read()
        encoded_string = base64.b64encode(mp3_data).decode('utf-8')

        completion = self.client.chat.completions.create(
            model="gpt-4o-mini-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "mp3"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        { 
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_string,
                                "format": "mp3"
                            }
                        }
                    ]
                },
            ]
        )
        return completion.choices[0].message.audio.transcript

class GLM4Audio:
    model_name = 'glm-4-voice'

    def __init__(self, api_key):
        self.api_key = api_key
        self.init_clinet()

    def init_clinet(self):
        self.client = ZhipuAI(api_key=self.api_key)
    
    def base64_encode_audio(self, audio_path):
        with open(audio_path, "rb") as audio_file:
            mp3_data = audio_file.read()
        return base64.b64encode(mp3_data).decode('utf-8')

    def chat(self, prompt, audio_path):
        audio_base64 = self.base64_encode_audio(audio_path)
        response = self.client.chat.completions.create(
            model="glm-4-voice",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_base64,
                                "format":"mp3"
                            }
                        }
                    ]
                },
            ],
            max_tokens=1024,
            stream=False, 
            temperature=0.01
        )   
        return response.choices[0].message.content
    
class Qwen1Audio:
    model_name = 'Qwen1Audio'

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.init_model()

    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="cuda:0", trust_remote_code=True, bf16=True).eval()
    
    def chat(self, prompt, audio_path):
        query = self.tokenizer.from_list_format([
            {'audio': audio_path}, 
            {'text': prompt},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None, temperature=0.01)
        return response
    
class Qwen2Audio:
    model_name = 'Qwen2Audio'
    
    def __init__(self, model_path=None, system='You are a helpful Assistant.'):
        self.model_path = model_path
        self.system = system
        self.init_model()

    def init_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")

    def chat(self, prompt, audio_path):
        conversation = [
            {'role': 'system', 'content': self.system}, 
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ]},
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        if "audio_url" in ele:
                            audios.append(librosa.load(BytesIO(urlopen(ele['audio_url']).read()), sr=self.processor.feature_extractor.sampling_rate)[0])
                        elif "audio" in ele:
                            audios.append(librosa.load(ele['audio'], sr=self.processor.feature_extractor.sampling_rate)[0])
        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True, sampling_rate=16000).to('cuda:0')
        inputs.input_ids = inputs.input_ids

        generate_ids = self.model.generate(**inputs, max_length=1024, temperature=0.01)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response




# Qwen2.5-Omni 音频推理类，接口与 Qwen2Audio 保持一致
class Qwen2OmniAudio:
    model_name = 'Qwen2.5-Omni-7B'

    def __init__(self, model_path=None, system='You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'):
        self.model_path = model_path
        self.system = system
        self.init_model()

    def init_model(self):
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(self.model_path, torch_dtype="auto", device_map="auto")

    def chat(self, prompt, audio_path):
        # 加载本地音频
        audio, sr = sf.read(audio_path)
        # 构造 conversation，支持文本+音频
        conversation = [
            {"role": "system", "content": [
                {"type": "text", "text": self.system}
            ]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt}
            ]},
        ]
        # 处理输入（兼容官方 utils，也支持自定义）
        if process_mm_info:
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        else:
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=text, audio=[audio], return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        # 推理
        text_ids, audio_out = self.model.generate(**inputs)
        response = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response
