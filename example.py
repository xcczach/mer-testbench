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


async def main():
    tasks = [
        test_bench.test_async(
            prediction_emotion_strs, true_emotion_strs, sample_ids, f"{i}"
        )
        for i in range(10)
    ]
    results = await asyncio.gather(*tasks)
    print(results)


asyncio.run(main())
