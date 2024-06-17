# python emo_tendency_test.py <prediction_file_name_without_csv>
# 读取数据
import pandas as pd
import sys

ground_truth_csv = pd.read_csv("data/check-openset.csv")
ground_truth_emotions = list(ground_truth_csv.iloc[:, 1].to_numpy())

prediction_csv = pd.read_csv(f"data/{sys.argv[1]}.csv")
prediction_emotions = list(prediction_csv.iloc[:, 1].to_numpy())

emotion_ids = list(ground_truth_csv.iloc[:, 0].to_numpy())
# 读取/生成ground truth/prediction情感倾向
import openai
import dotenv
from enum import Enum
import os

api_key = dotenv.get_key(".env", "API_KEY")
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.xi-ai.cn/v1",
)


class EmotionTendency(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


def convert_to_tendency(emotion_str: str) -> EmotionTendency:
    if "positive" in emotion_str:
        return EmotionTendency.POSITIVE
    elif "negative" in emotion_str:
        return EmotionTendency.NEGATIVE
    elif "neutral" in emotion_str:
        return EmotionTendency.NEUTRAL
    else:
        return EmotionTendency.UNKNOWN


def get_emotion_tendency(emotion_str: str) -> EmotionTendency:
    def get_response(emotion_str: str) -> str:
        prompt = f"根据给出的情绪列表，判断其情感倾向，回复positive/negative/neutral："
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt + emotion_str},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content

    return convert_to_tendency(get_response(emotion_str))


def get_emotion_tendencies(csv_path: str, emotions: list) -> list[EmotionTendency]:
    if os.path.exists(csv_path):
        emotion_tendency_dt = pd.read_csv(csv_path)
        emotion_tendencies = [
            convert_to_tendency(item) for item in emotion_tendency_dt["tendency"]
        ]
    else:
        emotion_tendencies = [
            get_emotion_tendency(emotion_str) for emotion_str in emotions
        ]
        emotion_tendency_dt = pd.DataFrame(
            {
                "emotion_id": emotion_ids,
                "emotions": emotions,
                "tendency": [item.value for item in emotion_tendencies],
            }
        )
        emotion_tendency_dt.to_csv(csv_path, index=False)
    return emotion_tendencies


ground_truth_tendencies = get_emotion_tendencies(
    "results/ground_truth_tendencies.csv", ground_truth_emotions
)
prediction_tendencies = get_emotion_tendencies(
    f"results/{sys.argv[1]}_tendencies.csv", prediction_emotions
)

# 列出情感倾向不匹配的sample
error_dt = pd.DataFrame(
    {
        "emotion_id": [],
        "ground_truth_emotions": [],
        "prediction_emotions": [],
        "ground_truth_tendency": [],
        "prediction_tendency": [],
    }
)
for index, sample_id in enumerate(emotion_ids):
    if ground_truth_tendencies[index] != prediction_tendencies[index]:
        error_dt = pd.concat(
            [
                error_dt,
                pd.DataFrame(
                    {
                        "emotion_id": [sample_id],
                        "ground_truth_emotions": [ground_truth_emotions[index]],
                        "prediction_emotions": [prediction_emotions[index]],
                        "ground_truth_tendency": [ground_truth_tendencies[index].value],
                        "prediction_tendency": [prediction_tendencies[index].value],
                    }
                ),
            ]
        )
error_dt.to_csv(f"results/{sys.argv[1]}_error.csv", index=False)
