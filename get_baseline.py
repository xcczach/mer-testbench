import json
import pandas as pd

result_df = pd.DataFrame(columns=["sample_id", "emotion_str"])
with open("result.json", "r", encoding="utf-8") as f:
    prediction_json = json.load(f)
for index, item in enumerate(prediction_json["data"]):
    result_df.loc[index] = [item["id"], item["emotion"]]
result_df.to_csv("baseline.csv", index=False)
