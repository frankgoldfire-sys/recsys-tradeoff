from pathlib import Path
import math

import pandas as pd


# ============================================================
# 1. 文件位置和实验设置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 10


# ============================================================
# 2. 读取训练集和验证集
# ============================================================

train = pd.read_csv(
    PROCESSED_DATA_DIR / "train.csv"
)

validation = pd.read_csv(
    PROCESSED_DATA_DIR / "validation.csv"
)


# ============================================================
# 3. 统计每部电影的流行度
# ============================================================

# 一部电影在训练集中获得4～5分的次数，
# 被定义为它的流行度
item_popularity_table = (
    train.groupby("movie_id")
    .size()
    .reset_index(name="popularity")
    .sort_values(
        ["popularity", "movie_id"],
        ascending=[False, True],
        kind="mergesort",
    )
)

ranked_movies = (
    item_popularity_table["movie_id"]
    .tolist()
)

item_popularity = (
    item_popularity_table
    .set_index("movie_id")["popularity"]
)


# ============================================================
# 4. 整理每位用户的历史与验证答案
# ============================================================

# 用户在训练集中已经喜欢过的电影
train_history = (
    train.groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

# 用户在验证集中喜欢的电影，即我们想预测的答案
validation_answers = (
    validation.groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)


# ============================================================
# 5. 定义热门电影和长尾电影
# ============================================================

# 流行度最高的20%电影定义为热门电影，
# 其余80%定义为长尾电影
number_of_head_movies = math.ceil(
    len(ranked_movies) * 0.20
)

head_movies = set(
    ranked_movies[:number_of_head_movies]
)


# ============================================================
# 6. 为每位用户生成前10名推荐
# ============================================================

user_results = []
all_recommended_movies = []

for user_id, relevant_movies in validation_answers.items():
    seen_movies = train_history.get(user_id, set())

    recommendations = []

    for movie_id in ranked_movies:
        # 不重复推荐用户训练阶段已经喜欢过的电影
        if movie_id not in seen_movies:
            recommendations.append(movie_id)

        if len(recommendations) == TOP_K:
            break

    all_recommended_movies.extend(recommendations)

    # --------------------------------------------------------
    # Recall@10 和 Hit Rate@10
    # --------------------------------------------------------

    hits = [
        movie_id
        for movie_id in recommendations
        if movie_id in relevant_movies
    ]

    recall_at_10 = (
        len(hits) / len(relevant_movies)
    )

    hit_rate_at_10 = int(
        len(hits) > 0
    )

    # --------------------------------------------------------
    # NDCG@10
    # 排名越靠前的正确推荐，获得的分数越高
    # --------------------------------------------------------

    dcg = 0.0

    for rank, movie_id in enumerate(
        recommendations,
        start=1,
    ):
        if movie_id in relevant_movies:
            dcg += 1 / math.log2(rank + 1)

    ideal_hit_count = min(
        len(relevant_movies),
        TOP_K,
    )

    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_hit_count + 1,
        )
    )

    ndcg_at_10 = (
        dcg / ideal_dcg
        if ideal_dcg > 0
        else 0.0
    )

    user_results.append(
        {
            "user_id": user_id,
            "recall_at_10": recall_at_10,
            "hit_rate_at_10": hit_rate_at_10,
            "ndcg_at_10": ndcg_at_10,
        }
    )


# ============================================================
# 7. 汇总总体结果
# ============================================================

user_results = pd.DataFrame(user_results)

unique_recommended_movies = set(
    all_recommended_movies
)

catalog_coverage = (
    len(unique_recommended_movies)
    / len(ranked_movies)
)

average_popularity = sum(
    item_popularity[movie_id]
    for movie_id in all_recommended_movies
) / len(all_recommended_movies)

long_tail_share = sum(
    movie_id not in head_movies
    for movie_id in all_recommended_movies
) / len(all_recommended_movies)


summary = pd.DataFrame(
    [
        {
            "model": "popularity",
            "top_k": TOP_K,
            "user_count": len(user_results),
            "recall_at_10": user_results[
                "recall_at_10"
            ].mean(),
            "hit_rate_at_10": user_results[
                "hit_rate_at_10"
            ].mean(),
            "ndcg_at_10": user_results[
                "ndcg_at_10"
            ].mean(),
            "catalog_coverage": catalog_coverage,
            "unique_recommended_movies": len(
                unique_recommended_movies
            ),
            "average_popularity": average_popularity,
            "long_tail_share": long_tail_share,
        }
    ]
)


# ============================================================
# 8. 保存并输出结果
# ============================================================

user_results.to_csv(
    RESULTS_DIR / "popularity_validation_per_user.csv",
    index=False,
)

summary.to_csv(
    RESULTS_DIR / "popularity_validation_summary.csv",
    index=False,
)

row = summary.iloc[0]

print("热门电影推荐基线完成")
print("=" * 50)

print(f"评价用户数：{int(row['user_count'])}")

print()
print("准确性指标：")
print(f"Recall@10：{row['recall_at_10']:.2%}")
print(f"Hit Rate@10：{row['hit_rate_at_10']:.2%}")
print(f"NDCG@10：{row['ndcg_at_10']:.2%}")

print()
print("分布指标：")
print(f"目录覆盖率：{row['catalog_coverage']:.2%}")
print(
    "被推荐过的不同电影："
    f"{int(row['unique_recommended_movies'])}"
    f"/{len(ranked_movies)}"
)
print(f"推荐电影平均流行度：{row['average_popularity']:.2f}")
print(f"长尾电影占比：{row['long_tail_share']:.2%}")

print()
print("结果保存至：")
print(RESULTS_DIR)