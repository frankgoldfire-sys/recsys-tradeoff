from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. 路径和实验设置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

RAW_DATA_DIR = (
    PROJECT_ROOT / "data" / "raw" / "ml-100k"
)

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TOP_K = 10

# 上一步验证集选择出的最佳参数
NEIGHBORS_PER_ITEM = 10

# ItemCF先产生100部候选电影，再做多样性重排序
CANDIDATE_POOL_SIZE = 100

# alpha越大，越重视ItemCF相关性
ALPHA_SETTINGS = [
    1.0,
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3,
    0.2,
    0.1,
    0.0,
]

# 最终模型允许的最大相对NDCG损失
MAX_RELATIVE_NDCG_DROP = 0.05


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
# 3. 读取电影类型信息
# ============================================================

genre_columns = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

movie_columns = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
] + genre_columns

movies = pd.read_csv(
    RAW_DATA_DIR / "u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    names=movie_columns,
)

movie_titles = (
    movies.set_index("movie_id")["title"]
    .to_dict()
)


# ============================================================
# 4. 构造用户—电影矩阵
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
# 5. 用户历史与验证集答案
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
# 6. 定义头部电影和长尾电影
# ============================================================

number_of_head_movies = math.ceil(
    len(ranked_movies) * 0.20
)

head_movies = set(
    ranked_movies[:number_of_head_movies]
)


# ============================================================
# 7. 构造保留10个近邻的ItemCF相似度矩阵
# ============================================================

base_similarity = cosine_similarity(
    user_item.T
)

np.fill_diagonal(
    base_similarity,
    0.0,
)

neighbor_order = np.argsort(
    -base_similarity,
    axis=1,
    kind="stable",
)

item_similarity = np.zeros_like(
    base_similarity
)

row_indices = np.arange(
    base_similarity.shape[0]
)[:, None]

top_neighbor_indices = neighbor_order[
    :,
    :NEIGHBORS_PER_ITEM,
]

item_similarity[
    row_indices,
    top_neighbor_indices,
] = base_similarity[
    row_indices,
    top_neighbor_indices,
]


# ============================================================
# 8. 构造电影类型差异矩阵
# ============================================================

movie_genres = (
    movies.set_index("movie_id")
    .reindex(movie_ids)[genre_columns]
    .fillna(0)
    .to_numpy(dtype="float32")
)

# 两部电影共同拥有的类型数量
genre_intersection = (
    movie_genres @ movie_genres.T
)

genre_count = movie_genres.sum(axis=1)

# 两部电影拥有的类型并集数量
genre_union = (
    genre_count[:, None]
    + genre_count[None, :]
    - genre_intersection
)

# Jaccard差异：
# 0表示类型完全相同
# 1表示没有共同类型
genre_dissimilarity = np.divide(
    genre_union - genre_intersection,
    genre_union,
    out=np.zeros_like(
        genre_union,
        dtype="float32",
    ),
    where=genre_union > 0,
)

np.fill_diagonal(
    genre_dissimilarity,
    0.0,
)


# ============================================================
# 9. 为每名用户生成ItemCF候选池
# ============================================================

print("正在生成ItemCF候选池……")

candidate_data = {}

for user_id in validation_answers:
    seen_movies = train_history.get(
        user_id,
        set(),
    )

    seen_indices = [
        movie_to_index[movie_id]
        for movie_id in sorted(seen_movies)
        if movie_id in movie_to_index
    ]

    # 每部候选电影与用户历史电影的相似度之和
    scores = item_similarity[
        :,
        seen_indices,
    ].sum(axis=1)

    candidate_movies = [
        movie_id
        for movie_id in movie_ids
        if movie_id not in seen_movies
    ]

    candidate_movies = sorted(
        candidate_movies,
        key=lambda movie_id: (
            -scores[
                movie_to_index[movie_id]
            ],
            -item_popularity[movie_id],
            movie_id,
        ),
    )[:CANDIDATE_POOL_SIZE]

    raw_scores = np.array(
        [
            scores[
                movie_to_index[movie_id]
            ]
            for movie_id in candidate_movies
        ],
        dtype="float32",
    )

    minimum_score = raw_scores.min()
    maximum_score = raw_scores.max()

    # 把ItemCF分数缩放至0～1，方便与类型差异结合
    if maximum_score > minimum_score:
        normalized_scores = (
            raw_scores - minimum_score
        ) / (
            maximum_score - minimum_score
        )
    else:
        normalized_scores = np.zeros_like(
            raw_scores
        )

    candidate_data[user_id] = {
        "movie_ids": candidate_movies,
        "raw_scores": raw_scores,
        "normalized_scores": normalized_scores,
    }


# ============================================================
# 10. 计算一张推荐列表的类型多样性
# ============================================================

def calculate_list_diversity(
    recommendations,
):
    pairwise_differences = []

    for first_position in range(
        len(recommendations)
    ):
        for second_position in range(
            first_position + 1,
            len(recommendations),
        ):
            first_movie = recommendations[
                first_position
            ]

            second_movie = recommendations[
                second_position
            ]

            first_index = movie_to_index[
                first_movie
            ]

            second_index = movie_to_index[
                second_movie
            ]

            difference = genre_dissimilarity[
                first_index,
                second_index,
            ]

            pairwise_differences.append(
                difference
            )

    if not pairwise_differences:
        return 0.0

    return float(
        np.mean(pairwise_differences)
    )


# ============================================================
# 11. 贪心式多样性重排序
# ============================================================

def rerank_candidates(
    candidate_movies,
    raw_scores,
    normalized_scores,
    alpha,
):
    raw_score_by_movie = {
        movie_id: float(raw_score)
        for movie_id, raw_score in zip(
            candidate_movies,
            raw_scores,
        )
    }

    relevance_by_movie = {
        movie_id: float(normalized_score)
        for movie_id, normalized_score in zip(
            candidate_movies,
            normalized_scores,
        )
    }

    selected_movies = []
    remaining_movies = list(
        candidate_movies
    )

    while (
        len(selected_movies) < TOP_K
        and remaining_movies
    ):

        def selection_key(movie_id):
            relevance_score = (
                relevance_by_movie[movie_id]
            )

            if not selected_movies:
                diversity_score = 0.0
            else:
                movie_index = movie_to_index[
                    movie_id
                ]

                diversity_score = float(
                    np.mean(
                        [
                            genre_dissimilarity[
                                movie_index,
                                movie_to_index[
                                    selected_movie
                                ],
                            ]
                            for selected_movie
                            in selected_movies
                        ]
                    )
                )

            combined_score = (
                alpha * relevance_score
                + (1 - alpha) * diversity_score
            )

            # 后面的值仅用于分数相同时稳定排序
            return (
                combined_score,
                raw_score_by_movie[movie_id],
                item_popularity[movie_id],
                -int(movie_id),
            )

        best_movie = max(
            remaining_movies,
            key=selection_key,
        )

        selected_movies.append(
            best_movie
        )

        remaining_movies.remove(
            best_movie
        )

    return selected_movies


# ============================================================
# 12. 评价一个alpha设置
# ============================================================

def evaluate_alpha(alpha):
    user_results = []
    recommendations_by_user = {}
    all_recommended_movies = []

    for user_id, relevant_movies in (
        validation_answers.items()
    ):
        current_candidates = (
            candidate_data[user_id]
        )

        recommendations = rerank_candidates(
            candidate_movies=current_candidates[
                "movie_ids"
            ],
            raw_scores=current_candidates[
                "raw_scores"
            ],
            normalized_scores=current_candidates[
                "normalized_scores"
            ],
            alpha=alpha,
        )

        recommendations_by_user[
            user_id
        ] = recommendations

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

        list_diversity = (
            calculate_list_diversity(
                recommendations
            )
        )

        user_results.append(
            {
                "user_id": user_id,
                "alpha": alpha,
                "recall_at_10": recall_at_10,
                "hit_rate_at_10": hit_rate_at_10,
                "ndcg_at_10": ndcg_at_10,
                "intra_list_diversity": (
                    list_diversity
                ),
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
        "alpha": alpha,
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
        "intra_list_diversity": user_results[
            "intra_list_diversity"
        ].mean(),
        "catalog_coverage": catalog_coverage,
        "unique_recommended_movies": len(
            unique_recommended_movies
        ),
        "average_popularity": (
            average_popularity
        ),
        "long_tail_share": long_tail_share,
    }

    return (
        summary,
        user_results,
        recommendations_by_user,
    )


# ============================================================
# 13. 运行全部alpha设置
# ============================================================

all_summaries = []
per_user_results = {}
all_recommendations = {}

for alpha in ALPHA_SETTINGS:
    print(
        f"正在运行：alpha = {alpha:.1f}"
    )

    (
        summary,
        user_result,
        recommendations,
    ) = evaluate_alpha(alpha)

    all_summaries.append(summary)
    per_user_results[alpha] = user_result
    all_recommendations[alpha] = (
        recommendations
    )

tuning_summary = pd.DataFrame(
    all_summaries
)


# ============================================================
# 14. 根据预先规定的规则选择最佳alpha
# ============================================================

base_ndcg = float(
    tuning_summary.loc[
        tuning_summary["alpha"] == 1.0,
        "ndcg_at_10",
    ].iloc[0]
)

minimum_acceptable_ndcg = (
    base_ndcg
    * (1 - MAX_RELATIVE_NDCG_DROP)
)

eligible_settings = tuning_summary[
    tuning_summary["ndcg_at_10"]
    >= minimum_acceptable_ndcg
].copy()

# 在NDCG下降不超过5%的设置中：
# 先选择列表多样性最高者；
# 若相同，再选NDCG更高者；
# 仍相同则选择alpha更高者。
best_row = (
    eligible_settings
    .sort_values(
        [
            "intra_list_diversity",
            "ndcg_at_10",
            "alpha",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
    .iloc[0]
)

best_alpha = float(
    best_row["alpha"]
)


# ============================================================
# 15. 保存汇总结果和逐用户结果
# ============================================================

tuning_summary.to_csv(
    RESULTS_DIR
    / "diversity_reranking_tuning_summary.csv",
    index=False,
)

per_user_results[
    best_alpha
].to_csv(
    RESULTS_DIR
    / "diversity_reranking_best_validation_per_user.csv",
    index=False,
)


# ============================================================
# 16. 保存最佳模型的推荐清单
# ============================================================

recommendation_rows = []

for user_id, recommendations in (
    all_recommendations[
        best_alpha
    ].items()
):
    for rank, movie_id in enumerate(
        recommendations,
        start=1,
    ):
        recommendation_rows.append(
            {
                "user_id": user_id,
                "rank": rank,
                "movie_id": movie_id,
                "title": movie_titles.get(
                    movie_id,
                    "Unknown title",
                ),
                "training_popularity": (
                    item_popularity[movie_id]
                ),
                "is_long_tail": (
                    movie_id
                    not in head_movies
                ),
            }
        )

pd.DataFrame(
    recommendation_rows
).to_csv(
    RESULTS_DIR
    / "diversity_reranking_best_recommendations.csv",
    index=False,
)


# ============================================================
# 17. 打印结果
# ============================================================

display_columns = [
    "alpha",
    "recall_at_10",
    "hit_rate_at_10",
    "ndcg_at_10",
    "intra_list_diversity",
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
    "intra_list_diversity",
    "catalog_coverage",
    "long_tail_share",
]

for column in percentage_columns:
    display_result[column] = (
        display_result[column]
        .map(
            lambda value: f"{value:.2%}"
        )
    )

print()
print("=" * 90)
print("ItemCF多样性重排序参数比较")
print("=" * 90)

print(
    display_result.to_string(
        index=False
    )
)

print()
print("模型选择规则：")
print(
    "在NDCG@10相对下降不超过5%的设置中，"
    "选择列表多样性最高的alpha。"
)

print()
print(
    "原始ItemCF的NDCG@10："
    f"{base_ndcg:.2%}"
)

print(
    "允许的最低NDCG@10："
    f"{minimum_acceptable_ndcg:.2%}"
)

print(
    "最终选择的alpha："
    f"{best_alpha:.1f}"
)

print()
print("结果保存至：")
print(
    RESULTS_DIR
    / "diversity_reranking_tuning_summary.csv"
)
print(
    RESULTS_DIR
    / "diversity_reranking_best_validation_per_user.csv"
)
print(
    RESULTS_DIR
    / "diversity_reranking_best_recommendations.csv"
)