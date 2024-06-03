import dotenv
import openai
import pandas as pd
import sys
import ast

api_key = dotenv.get_key(".env", "API_KEY")
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.xi-ai.cn/v1",
)
prompt = "翻译为英文小写："


def translation(text: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{prompt}\n{text}"}],
        temperature=0.1,
    )
    return completion.choices[0].message.content


def list_translation(text_str: str) -> str:
    try:
        text_list = ast.literal_eval(text_str)
        return [translation(text) for text in text_list]
    except:
        return text_str


file_name = sys.argv[1]
dt = pd.read_csv(f"{file_name}.csv")
dt["emotion_str"] = dt["emotion_str"].apply(translation)
dt.to_csv(f"{file_name}_translated.csv", index=False)
