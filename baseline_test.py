import test_bench
import json
import pandas as pd
import asyncio

prediction_emotion_strs = []
sample_ids = []
with open("result.json", "r", encoding="utf-8") as f:
    prediction_json = json.load(f)
for item in prediction_json["data"]:
    prediction_emotion_strs.append(item["emotion"])
    sample_ids.append(item["id"])

df = pd.read_csv("final-openset-chinese.csv")
true_emotion_strs = list(df.iloc[:, 1].to_numpy())


eval_rounds = 10
temperature = 0.1


async def main():
    tasks = [
        test_bench.test_async(
            prediction_emotion_strs, true_emotion_strs, sample_ids, f"{i}", temperature
        )
        for i in range(eval_rounds)
    ]

    results = await asyncio.gather(*tasks)
    results_str = f"temperature: {temperature}\n"
    for index, item in enumerate(results):
        result_str = f"round: {index}, accuracy: {item[1][0]}, recall: {item[1][1]}, score: {item[1][2]}"
        results_str += result_str + "\n"
        print(result_str)
        item[0].to_csv(f"eval_result_baseline_r{index}.csv", index=False)
    accuracys, recalls, scores = zip(*[item[1] for item in results])
    confidence = 0.95
    mean_accuracy, margin_of_error_accuracy = test_bench.get_mean_and_error(
        accuracys, confidence
    )
    mean_recall, margin_of_error_recall = test_bench.get_mean_and_error(
        recalls, confidence
    )
    mean_score, margin_of_error_score = test_bench.get_mean_and_error(
        scores, confidence
    )
    mean_result_str = f"accurary: {mean_accuracy} +- {margin_of_error_accuracy}, recall: {mean_recall} +- {margin_of_error_recall}, score: {mean_score} +- {margin_of_error_score}, confidence: {confidence}"
    print(mean_result_str)
    results_str += mean_result_str + "\n"
    with open("eval_results_baseline.txt", "w") as f:
        f.write(results_str)


asyncio.run(main())
