from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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
# 3. 建立“用户—电影”矩阵
# ============================================================

# 行代表用户，列代表电影。
# 数值1表示该用户在训练集中喜欢这部电影，
# 数值0表示训练集中没有这条喜欢记录。
user_item = pd.crosstab(
    train["user_id"],
    train["movie_id"],
).astype("float32")

movie_ids = user_item.columns.to_numpy()

# 建立“电影编号 → 矩阵位置”的对应关系
movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}


# ============================================================
# 4. 计算电影流行度
# ============================================================

item_popularity = (
    user_item.sum(axis=0)
    .astype(int)
)

# 先按照流行度从高到低排序；
# 如果流行度相同，则按照电影编号从小到大排序。
ranked_movies = sorted(
    movie_ids,
    key=lambda movie_id: (
        -item_popularity[movie_id],
        movie_id,
    ),
)


# ============================================================
# 5. 计算电影之间的余弦相似度
# ============================================================

# user_item.T 的每一行代表一部电影，
# 每一列代表一位用户。
#
# 如果两部电影经常被同一批用户喜欢，
# 它们的余弦相似度就会更高。
similarity = cosine_similarity(
    user_item.T
)

# 一部电影与自己的相似度本来是1。
# 推荐时不需要使用“自己与自己”的相似度，
# 所以把矩阵对角线设置为0。
np.fill_diagonal(
    similarity,
    0.0,
)


# ============================================================
# 6. 整理用户历史和验证集答案
# ============================================================

# 每位用户在训练集中已经喜欢过的电影
train_history = (
    train.groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

# 每位用户在验证集中喜欢的电影
validation_answers = (
    validation.groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)


# ============================================================
# 7. 定义热门电影和长尾电影
# ============================================================

# 流行度最高的20%电影定义为热门电影，
# 剩余80%定义为长尾电影。
number_of_head_movies = math.ceil(
    len(ranked_movies) * 0.20
)

head_movies = set(
    ranked_movies[:number_of_head_movies]
)


# ============================================================
# 8. 为每位用户生成前10名推荐
# ============================================================

user_results = []
all_recommended_movies = []

for user_id, relevant_movies in validation_answers.items():
    seen_movies = train_history.get(
        user_id,
        set(),
    )

    # 找到该用户喜欢过的电影在相似度矩阵中的位置
    seen_indices = [
        movie_to_index[movie_id]
        for movie_id in sorted(seen_movies)
        if movie_id in movie_to_index
    ]

    # 对于每一部候选电影：
    # 把它与用户历史电影的相似度相加，
    # 得到这部电影对该用户的推荐分数。
    scores = similarity[
        :,
        seen_indices,
    ].sum(axis=1)

    # 只考虑用户训练阶段没有喜欢过的电影
    candidate_movies = [
        movie_id
        for movie_id in movie_ids
        if movie_id not in seen_movies
    ]

    # 排序规则：
    # 1. 推荐分数越高越靠前；
    # 2. 分数相同，流行度越高越靠前；
    # 3. 仍然相同，电影编号越小越靠前。
    recommendations = sorted(
        candidate_movies,
        key=lambda movie_id: (
            -scores[
                movie_to_index[movie_id]
            ],
            -item_popularity[movie_id],
            movie_id,
        ),
    )[:TOP_K]

    all_recommended_movies.extend(
        recommendations
    )

    # --------------------------------------------------------
    # Recall@10 和 Hit Rate@10
    # --------------------------------------------------------

    hits = [
        movie_id
        for movie_id in recommendations
        if movie_id in relevant_movies
    ]

    recall_at_10 = (
        len(hits)
        / len(relevant_movies)
    )

    hit_rate_at_10 = int(
        len(hits) > 0
    )

    # --------------------------------------------------------
    # NDCG@10
    # 正确电影排得越靠前，得分越高
    # --------------------------------------------------------

    dcg = 0.0

    for rank, movie_id in enumerate(
        recommendations,
        start=1,
    ):
        if movie_id in relevant_movies:
            dcg += (
                1
                / math.log2(rank + 1)
            )

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
# 9. 计算总体指标
# ============================================================

user_results = pd.DataFrame(
    user_results
)

unique_recommended_movies = set(
    all_recommended_movies
)

catalog_coverage = (
    len(unique_recommended_movies)
    / len(movie_ids)
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
            "model": "itemcf_cosine",
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
# 10. 保存结果
# ============================================================

user_results.to_csv(
    RESULTS_DIR / "itemcf_validation_per_user.csv",
    index=False,
)

summary.to_csv(
    RESULTS_DIR / "itemcf_validation_summary.csv",
    index=False,
)


# ============================================================
# 11. 在终端显示结果
# ============================================================

row = summary.iloc[0]

print("物品协同过滤完成")
print("=" * 50)

print(f"用户—电影矩阵：{user_item.shape}")
print(f"电影相似度矩阵：{similarity.shape}")
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
    f"/{len(movie_ids)}"
)
print(
    "推荐电影平均流行度："
    f"{row['average_popularity']:.2f}"
)
print(
    "长尾电影占比："
    f"{row['long_tail_share']:.2%}"
)

print()
print("结果保存至：")
print(RESULTS_DIR)