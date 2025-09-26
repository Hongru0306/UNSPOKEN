DIRECT_SINGLE_ZH_KEY = """
请尝试根据提供的音频与提示回答下面的单答案选择题。
提示:关键隐喻之处在于: {key}
{question}
{choices}
所以答案是:
"""

DIRECT_SINGLE_EN_KEY= """
Please try to answer the single-answer multiple choice question below based on the audio and hint provided.
Hint: The key point of the metaphor is: {key}
{question}
{choices}
So the answer is:
"""


DIRECT_MULTIPLE_ZH_KEY = """
请尝试根据提供的音频与提示回答下面的多答案选择题。
提示:关键隐喻之处在于: {key}
{question}
{choices}
所以答案是:
"""

DIRECT_MULTIPLE_EN_KEY = """
Please try to answer the multiple-answer multiple choice question below based on the audio and hint provided.
Hint: The key point of the metaphor is: {key}
{question}
{choices}
So the answer is:
"""

DIRECT_SINGLE_ZH_text = """
请尝试根据提供的文本回答下面的单答案选择题。
{text}
{question}
{choices}
所以答案是:
"""

DIRECT_SINGLE_EN_text = """
Please try to answer the single-answer multiple choice question below based on the content provided.
{text}
{question}
{choices}
So the answer is:
"""

DIRECT_MULTIPLE_ZH_text = """
请尝试根据提供的内容回答下面的多答案选择题。
{text}
{question}
{choices}
所以答案是:
"""
DIRECT_MULTIPLE_EN_text = """
Please try to answer the multiple-answer multiple choice question below based on the content provided.
{text}
{question}
{choices}
So the answer is:
"""


DIRECT_SINGLE_ZH = """
请尝试根据提供的音频回答下面的单答案选择题。
{question}
{choices}
所以答案是:
"""

DIRECT_SINGLE_EN = """
Please try to answer the single-answer multiple choice question below based on the audio provided.
{question}
{choices}
So the answer is:
"""

DIRECT_MULTIPLE_ZH = """
请尝试根据提供的音频回答下面的多答案选择题。
{question}
{choices}
所以答案是:
"""

DIRECT_MULTIPLE_EN = """
Please try to answer the multiple-answer multiple choice question below based on the audio provided.
{question}
{choices}
So the answer is:
"""

COT_SINGLE_ZH = """
请尝试根据提供的音频回答下面的单答案选择题。
让我们仔细考虑一下每个选项。让我们一步一步来思考。
{question}
{choices}
"""

COT_SINGLE_EN = """
Please try to answer the single-answer multiple choice question below based on the audio provided.
Let's think through each option. Let's think step by step.
{question}
{choices}
"""

COT_MULTIPLE_ZH = """
请尝试根据提供的音频回答下面的多答案选择题。
让我们仔细考虑一下每个选项。让我们一步一步来思考。
{question}
{choices}
"""

COT_MULTIPLE_EN = """
Please try to answer the multiple-answer multiple choice question below based on the audio provided.
Let's think through each option. Let's think step by step.
{question}
{choices}
"""


XLT_SINGLE_EN = """
I want you to act as an audio reasoning expert for Chinese.
Question: Single choice, Which of the following best illustrates the underlying joke in the story?
{question}
{choices}
You should retell the Question in English.
You should do step-by-step answer to obtain a option answer .
You should step-by-step answer the question.
"""

XLT_SINGLE_ZH = """
我希望你能充当一名英文音频推理专家。
问题：单选，以下哪一项最能体现该段故事中的隐含笑点？
{question}
{choices}
你应该用英文重新叙述问题。
你应该逐步回答问题以获得一个最佳选项答案。
你应该逐步回答问题。
"""

XLT_MULTIPLE_EN = """
I want you to act as an audio reasoning expert for Chinese.
Question: Multiple choice, Which of the following best illustrates the underlying joke in the story?
{question}
{choices}
You should retell the Question in English.
You should do step-by-step answer to obtain multiple option answers.
You should step-by-step answer the question.
"""

XLT_MULTIPLE_ZH = """
我希望你能充当一名英文音频推理专家。
问题：多选，以下哪几项最能体现该段故事中的隐含笑点？
{question}
{choices}
你应该用英文重新叙述问题。
你应该逐步回答问题以获得多个最佳选项答案。
你应该逐步回答问题。
"""