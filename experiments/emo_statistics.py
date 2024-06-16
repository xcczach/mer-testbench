# python emo_statistics.py <prediction_file_name_without_csv>
import pandas as pd
import sys
import matplotlib.pyplot as plt
import utils
import numpy as np

# 读取文件
file_prefix = sys.argv[1]
dt = pd.read_csv(f"data/{file_prefix}.csv")

# 统计每种情绪出现次数
all_emotions = [
    utils.emotion_str_to_list(emotion_str) for emotion_str in list(dt["emotion_str"])
]
all_emotions = [item for sublist in all_emotions for item in sublist]
emotion_count = {}
for emotion in all_emotions:
    if emotion not in emotion_count:
        emotion_count[emotion] = 1
    else:
        emotion_count[emotion] += 1

# 画图 topx情绪分布
emotion_count = sorted(emotion_count.items(), key=lambda x: x[1], reverse=True)
top_x = 10
plt.barh(
    [x[0] for x in emotion_count[:top_x]],
    [x[1] for x in emotion_count[:top_x]],
)
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.title(f"Emotion Distribution of {file_prefix}, top{top_x}")
plt.tick_params(axis="x", labelsize=8)
plt.savefig(f"results/{file_prefix}_top{top_x}_emotions.png")
# 画图 情绪出现次数分布
plt.clf()
plt.hist([x[1] for x in emotion_count], bins=50)
plt.xlabel("Count")
plt.ylabel("Frequency")
plt.title(f"Emotion Count Distribution of {file_prefix}")
plt.savefig(f"results/{file_prefix}_emotion_count_distribution.png")

# 其他统计数据：情绪数量、情绪数量平均值、情绪数量中位数、情绪数量标准差
with open(f"results/{file_prefix}_emo_statistics.txt", "w", encoding="utf-8") as f:
    f.write(f"情绪总数：{len(emotion_count)}\n")
    f.write(
        f"情绪出现次数平均值：{sum([x[1] for x in emotion_count]) / len(emotion_count)}\n"
    )
    f.write(f"情绪出现次数中位数：{emotion_count[len(emotion_count) // 2][1]}\n")
    f.write(f"情绪出现次数标准差：{np.std([x[1] for x in emotion_count])}\n")
