# 🎧 UNSPOKEN: Can Audio Language Models Listen Between the Lines?

[](https://arxiv.org/abs/XXXX.XXXXX)
[](https://unspoken-demo.vercel.app)
[](https://python.org)
[](LICENSE)

<div align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-inproceedings-B31B1B?logo=arxiv&logoColor=white" />
  </a>
  <a href="https://github.com/Hongru0306/UNSPOKEN" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Unspoken-181717?logo=github&logoColor=white" />
  </a>
  <a href="https://huggingface.co/你的模型路径" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-Model-ffc107?logo=huggingface&logoColor=white" />
  </a>
</div>


<div align="center">

[![OpenAPI](https://img.shields.io/badge/OpenAPI-6BA539?logo=openapiinitiative&logoColor=white)](#)
[![ChatGPT](https://img.shields.io/badge/ChatGPT-74aa9c?logo=openai&logoColor=white)](#)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-886FBF?logo=googlegemini&logoColor=fff)](#)
<a href="#" target="_blank">
  <span style="background: #fff; border-radius: 6px; padding: 2px 8px; display: inline-flex; align-items: center; border: 0.0px solid #ddd;">
    <img src="./_assets/qwen2audio.png" alt="Qwen2.5-Omni" height="18"/>
  </span>
</a>
<a href="#" target="_blank">
  <span style="background: #fff; border-radius: 6px; padding: 2px 8px; display: inline-flex; align-items: center; border: 0.0px solid #ddd;">
    <img src="./_assets/qwen-omni.png" alt="Qwen2.5-Omni" height="18"/>
  </span>
</a>
[![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=fff)](#)
	[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#)
  [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)

</div>



> *"The real meaning is often hidden in how something is said, not just what is said."*

**UNSPOKEN** is the first **bilingual** (Chinese-English) benchmark designed to evaluate *metaphorical reasoning capabilities* in **Audio Language Models (ALMs)**. Unlike traditional transcription-based evaluations, **UNSPOKEN** challenges models to understand non-literal language by *leveraging subtle acoustic cues like prosody, emotional inflection, and phonetic ambiguity*.

<!-- <div align="center">
  <video src="./_assets/demo.mp4" controls width="1000">
    您的浏览器不支持 video 标签。
  </video>
</div> -->


## 🚀 What's New?

Current **ALMs** excel at literal speech understanding but struggle with the nuanced world of *metaphors, irony, and cultural references*. **UNSPOKEN** reveals that even state-of-the-art models achieve only **68.9% accuracy** - still significantly below the human average of **80.9%**, though.


### ✨ Key Features

- **🎯 Audio-Centric Evaluation**: Grounded in actual audio, not just transcriptions
- **🌍 Bilingual Coverage**: 2,764 validated QA pairs in Chinese and English
- **🧠 Multi-Dimensional Reasoning**: Semantic, acoustic, and contextual understanding
- **📊 Fine-Grained Categories**: 6 metaphor types (Puns, Cultural Metaphors, Irony, etc.)
- **⚡ Easy Integration**: Simple API for evaluating your own ALMs

### 📊 Dataset Overview

<div align="center">

| Metric | Value |
|--------|-------|
| **Total QA Pairs** | 2,764 |
| **Audio Segments** | 1,382 |
| **Total Duration** | ~38 hours |
| **Languages** | Chinese & English |
| **Question Types** | Single & Multiple Choice |

</div>

## 🚀 Quick Start and Installation

```bash
git clone https://github.com/Hongru0306/UNSPOKEN.git
cd unspoken
pip install -r requirements.txt
```
Alternatively, you can setup environements with `conda`:
```bash
conda create --name <env_name> --file environment.txt
```

The following code snippet demonstartes the basic evaluation setup:
```python
from model import Qwen2OmniAudio
from utils import run_experiment
import pandas as pd
## Load your model
model = Qwen2OmniAudio(model_path='your-model-path')
## Load dataset
df = pd.read_csv('./final_utf8.csv')
## Run evaluation
run_experiment(model, df, task='direct', path='./sliced_mp3')

### Advanced Prompting Strategies
from prompt import DIRECT_SINGLE_EN, COT_SINGLE_EN, XLT_SINGLE_EN
"""
Three prompting strategies available:
- Direct: Standard question-answering
- Chain-of-Thought: Step-by-step reasoning
- Cross-Lingual Transfer: Language-switching prompts
"""
```
Or run your evaluation with a single `bash` command:
```bash
python main.py --model Qwen2OmniAudio --model_path '' --task direct --input ./final_utf8.csv --audio_path ./sliced_mp3
```




## Citation
```
@inproceedings{xiao2025unspoken,
  title     = {Can Audio Language Models Listen Between the Lines? A Study on Metaphorical Reasoning via Unspoken},
  author    = {Xiao, Hongru and Li, Xiang and Pan, Duyi and Zhang, Longfei and Song, Zhixue and Han, Jiale and Lai, Songning and Chen, Wenshuo and Tang, Jing and Wang, Benyou},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)},
  year      = {2025},
  isbn      = {979-8-4007-2035-2/2025/10},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3746027.3758173},
  location  = {Dublin, Ireland},
  series    = {MM '25}
}
```
