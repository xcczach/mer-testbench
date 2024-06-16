import ast
import openai
import dotenv

api_key = dotenv.get_key(".env", "API_KEY")
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.xi-ai.cn/v1",
)


def emotion_str_to_list(emotion_str: str) -> list[str]:
    try:
        return ast.literal_eval(emotion_str)
    except:
        prompt = "格式：['happy','sad','excited'], 将以下内容转化为Python列表："
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt + emotion_str},
            ],
            temperature=0.1,
        )
        return ast.literal_eval(response.choices[0].message.content)
