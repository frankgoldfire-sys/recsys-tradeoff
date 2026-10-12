from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. 路径和实验设置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 10

NEIGHBOR_SETTINGS = [
    10,
    20,
    40,
    80,
    160,
    None,  # None表示保留全部相似关系
]


# ============================================================
# 2. 读取数据
# ============================================================

train = pd.read_csv(
    DATA_DIR / "train.csv"
)

validation = pd.read_csv(
    DATA_DIR / "validation.csv"
)


# ============================================================
# 3. 用户—电影矩阵
# ============================================================

user_item = pd.crosstab(
    train["user_id"],
    train["movie_id"],
).astype("float32")

movie_ids = user_item.columns.to_numpy()

movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}

item_popularity = (
    user_item.sum(axis=0)
    .astype(int)
)

ranked_movies = sorted(
    movie_ids,
    key=lambda movie_id: (
        -item_popularity[movie_id],
        movie_id,
    ),
)


# ============================================================
# 4. 用户历史与验证答案
# ============================================================

train_history = (
    train.groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

validation_answers = (
    validation.groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)


# ============================================================
# 5. 热门电影与长尾电影
# ============================================================

number_of_head_movies = math.ceil(
    len(ranked_movies) * 0.20
)

head_movies = set(
    ranked_movies[:number_of_head_movies]
)


# ============================================================
# 6. 原始电影相似度矩阵
# ============================================================

base_similarity = cosine_similarity(
    user_item.T
)

np.fill_diagonal(
    base_similarity,
    0.0,
)

# 每一行中的电影相似度从高到低排列。
# 相似度相同时保持电影编号顺序，保证结果可复现。
neighbor_order = np.argsort(
    -base_similarity,
    axis=1,
    kind="stable",
)


# ============================================================
# 7. 构造只保留前N个近邻的相似度矩阵
# ============================================================

def keep_top_neighbors(
    similarity,
    number_of_neighbors,
):
    if number_of_neighbors is None:
        return similarity.copy()

    pruned_similarity = np.zeros_like(
        similarity
    )

    row_indices = np.arange(
        similarity.shape[0]
    )[:, None]

    top_neighbor_indices = neighbor_order[
        :,
        :number_of_neighbors,
    ]

    pruned_similarity[
        row_indices,
        top_neighbor_indices,
    ] = similarity[
        row_indices,
        top_neighbor_indices,
    ]

    return pruned_similarity


# ============================================================
# 8. 评价一个相似度矩阵
# ============================================================

def evaluate_model(
    current_similarity,
    setting_name,
):
    user_results = []
    all_recommended_movies = []

    for user_id, relevant_movies in validation_answers.items():
        seen_movies = train_history.get(
            user_id,
            set(),
        )

        seen_indices = [
            movie_to_index[movie_id]
            for movie_id in sorted(seen_movies)
            if movie_id in movie_to_index
        ]

        scores = current_similarity[
            :,
            seen_indices,
        ].sum(axis=1)

        candidate_movies = [
            movie_id
            for movie_id in movie_ids
            if movie_id not in seen_movies
        ]

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

    summary = {
        "neighbors_per_item": setting_name,
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

    return summary, user_results


# ============================================================
# 9. 运行所有参数
# ============================================================

all_summaries = []
per_user_results = {}

for number_of_neighbors in NEIGHBOR_SETTINGS:
    setting_name = (
        "all"
        if number_of_neighbors is None
        else str(number_of_neighbors)
    )

    print(
        "正在运行："
        f"每部电影保留 {setting_name} 个近邻"
    )

    current_similarity = keep_top_neighbors(
        base_similarity,
        number_of_neighbors,
    )

    summary, user_result = evaluate_model(
        current_similarity,
        setting_name,
    )

    all_summaries.append(summary)
    per_user_results[setting_name] = user_result


# ============================================================
# 10. 按NDCG@10选择最佳参数
# ============================================================

tuning_summary = pd.DataFrame(
    all_summaries
)

# 参数相同表现时，优先选择近邻数量更少的简单模型
selection_order = {
    "10": 10,
    "20": 20,
    "40": 40,
    "80": 80,
    "160": 160,
    "all": len(movie_ids),
}

tuning_summary["selection_order"] = (
    tuning_summary["neighbors_per_item"]
    .map(selection_order)
)

best_row = (
    tuning_summary
    .sort_values(
        [
            "ndcg_at_10",
            "selection_order",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .iloc[0]
)

best_setting = best_row[
    "neighbors_per_item"
]


# ============================================================
# 11. 保存结果
# ============================================================

tuning_summary.drop(
    columns="selection_order"
).to_csv(
    RESULTS_DIR / "itemcf_tuning_summary.csv",
    index=False,
)

per_user_results[
    best_setting
].to_csv(
    RESULTS_DIR / "itemcf_tuned_validation_per_user.csv",
    index=False,
)


# ============================================================
# 12. 打印参数比较
# ============================================================

display_columns = [
    "neighbors_per_item",
    "recall_at_10",
    "hit_rate_at_10",
    "ndcg_at_10",
    "catalog_coverage",
    "long_tail_share",
]

display_result = tuning_summary[
    display_columns
].copy()

percentage_columns = [
    "recall_at_10",
    "hit_rate_at_10",
    "ndcg_at_10",
    "catalog_coverage",
    "long_tail_share",
]

for column in percentage_columns:
    display_result[column] = (
        display_result[column]
        .map(lambda value: f"{value:.2%}")
    )

print()
print("=" * 70)
print("ItemCF参数比较")
print("=" * 70)

print(
    display_result.to_string(
        index=False
    )
)

print()
print("根据验证集NDCG@10选择的最佳参数：")
print(
    f"每部电影保留 {best_setting} 个近邻"
)

print()
print("结果保存至：")
print(
    RESULTS_DIR
    / "itemcf_tuning_summary.csv"
)