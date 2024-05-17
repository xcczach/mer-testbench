import pandas as pd
import openai
import ast
import asyncio
import dotenv
import scipy.stats as stats
import numpy as np

api_key = dotenv.get_key(".env", "API_KEY")
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.xi-ai.cn/v1",
)
prompt = "请扮演情感领域的专家。我们将提供一组情绪。请将这些情绪分组，每组包含同义词或一致的情感术语。直接以Python列表形式输出结果。"


def get_emotion_ids(
    prediction_emotions: list[str], true_emotions: list[str]
) -> tuple[set[int], set[int]]:

    all_emotions = set(prediction_emotions + true_emotions)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{prompt}\n{all_emotions}"}],
    )

    arrs = ast.literal_eval(completion.choices[0].message.content)
    prediction_ids = set()
    true_ids = set()
    for emotion in prediction_emotions:
        for i, arr in enumerate(arrs):
            if emotion in arr:
                prediction_ids.add(i)
    for emotion in true_emotions:
        for i, arr in enumerate(arrs):
            if emotion in arr:
                true_ids.add(i)
    return prediction_ids, true_ids, arrs


def get_accuracy(prediction_ids: set[int], true_ids: set[int]):
    return len(prediction_ids.intersection(true_ids)) / len(prediction_ids)


def get_recall(prediction_ids: set[int], true_ids: set[int]):
    return len(prediction_ids.intersection(true_ids)) / len(true_ids)


def test(
    prediction_emotion_strs: list[str],
    true_emotion_strs: list[str],
    sample_ids: list[str] | None = None,
    tag: str = "default",
) -> tuple[pd.DataFrame, tuple[float, float, float]]:
    """
    prediction_emotion_strs/ true_emotion_strs: e.g. ["['喜悦', '高兴']", "['悲伤', '难过']"]
    sample_ids: e.g. ["sample_00000007", "sample_00000021"], should align with prediction_emotion_strs
    tag: for logging purpose e.g. "default"
    """
    result_df = pd.DataFrame(
        columns=[
            "sample_index",
            "sample",
            "emotion_group",
            "accuracy",
            "recall",
            "score",
        ]
    )
    if sample_ids is None:
        result_df = result_df.drop(columns=["sample"])

    for i in range(len(prediction_emotion_strs)):
        try:
            prediction_emotions = ast.literal_eval(prediction_emotion_strs[i])
            true_emotions = ast.literal_eval(true_emotion_strs[i])

            prediction_ids, true_ids, emotion_group = get_emotion_ids(
                prediction_emotions, true_emotions
            )
            sample_index = i
            accuracy = get_accuracy(prediction_ids, true_ids)
            recall = get_recall(prediction_ids, true_ids)
            score = (accuracy + recall) / 2
            result_df.loc[i] = (
                [
                    sample_index,
                    sample_ids[i],
                    emotion_group,
                    accuracy,
                    recall,
                    score,
                ]
                if sample_ids is not None
                else [
                    sample_index,
                    emotion_group,
                    accuracy,
                    recall,
                    score,
                ]
            )
        except:
            print(f"WARNING at {tag}: sample {i} failed")
            continue

    mean_accuracy = result_df["accuracy"].mean()
    mean_recall = result_df["recall"].mean()
    mean_score = result_df["score"].mean()
    return result_df, (mean_accuracy, mean_recall, mean_score)


async def test_async(
    prediction_emotion_strs: list[str],
    true_emotion_strs: list[str],
    sample_ids: list[str] | None = None,
    tag: str = "default",
) -> tuple[pd.DataFrame, tuple[float, float, float]]:
    return await asyncio.get_event_loop().run_in_executor(
        None, test, prediction_emotion_strs, true_emotion_strs, sample_ids, tag
    )


def get_mean_and_error(
    data: list[float], confidence_level: float = 0.95
) -> tuple[float, float]:
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    n = len(data)
    degrees_of_freedom = n - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
    standard_error = std_dev / np.sqrt(n)
    margin_of_error = t_critical * standard_error
    return mean, margin_of_error
