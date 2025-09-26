from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from transformers.generation import GenerationConfig
import torch
import base64
from zhipuai import ZhipuAI
from io import BytesIO
from urllib.request import urlopen
import librosa
from openai import OpenAI
from google import genai

import warnings
warnings.filterwarnings("ignore")

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
