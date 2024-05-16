import pandas as pd
import openai
import ast
import json


client = openai.OpenAI(
    api_key="sk-SPmgatijZMoa4g4WE9Ea5c9d0a124d4dBb43418b88C0Ae69",
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
    # print(arrs)
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


if __name__ == "__main__":
    prediction_emotion_strs = []
    with open("result.json", "r", encoding="utf-8") as f:
        prediction_json = json.load(f)
    for item in prediction_json["data"]:
        prediction_emotion_strs.append(item["emotion"])

    df = pd.read_csv("final-openset-chinese.csv")
    true_emotion_strs = list(df.iloc[:, 1].to_numpy())
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

    for i in range(len(prediction_emotion_strs)):
        try:
            prediction_emotions = ast.literal_eval(prediction_emotion_strs[i])
            true_emotions = ast.literal_eval(true_emotion_strs[i])

            prediction_ids, true_ids, emotion_group = get_emotion_ids(
                prediction_emotions, true_emotions
            )
            sample_index = i
            sample = prediction_json["data"][i]["id"]
            accuracy = get_accuracy(prediction_ids, true_ids)
            recall = get_recall(prediction_ids, true_ids)
            score = (accuracy + recall) / 2
            result_df.loc[i] = [
                sample_index,
                sample,
                emotion_group,
                accuracy,
                recall,
                score,
            ]
        except:
            print(f"sample {i} failed")
            continue
        print(f"sample {i}: accuracy: {accuracy}, recall: {recall}, score: {score}")

    result_df.to_csv("eval_result.csv", index=False)
    # get average
    print("Accuracy", result_df["accuracy"].mean())
    print("Recall", result_df["recall"].mean())
    print("Score", result_df["score"].mean())
