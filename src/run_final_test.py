from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. 路径与已经锁定的实验参数
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

# 以下参数来自验证集，不再修改
NEIGHBORS_PER_ITEM = 10
CANDIDATE_POOL_SIZE = 100
SELECTED_ALPHA = 0.4


# ============================================================
# 2. 读取数据
# ============================================================

train = pd.read_csv(
    PROCESSED_DATA_DIR / "train.csv"
)

validation = pd.read_csv(
    PROCESSED_DATA_DIR / "validation.csv"
)

test = pd.read_csv(
    PROCESSED_DATA_DIR / "test.csv"
)

# 调参已经结束，因此将训练集和验证集合并，
# 用它们训练最终模型
final_train = pd.concat(
    [train, validation],
    ignore_index=True,
)

print("最终测试开始")
print("=" * 70)
print(f"原训练集记录：{len(train)}")
print(f"验证集记录：{len(validation)}")
print(f"最终训练记录：{len(final_train)}")
print(f"测试集记录：{len(test)}")
print()


# ============================================================
# 3. 读取电影名称和类型
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
# 4. 构造最终用户—电影矩阵
# ============================================================

user_item = pd.crosstab(
    final_train["user_id"],
    final_train["movie_id"],
).astype("float32")

movie_ids = user_item.columns.to_numpy()
movie_id_set = set(movie_ids)

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
# 5. 最终训练历史和测试答案
# ============================================================

train_history = (
    final_train.groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

test_answers = (
    test.groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

cold_test_records = int(
    (~test["movie_id"].isin(movie_id_set)).sum()
)

cold_test_movies = set(
    test.loc[
        ~test["movie_id"].isin(movie_id_set),
        "movie_id",
    ]
)

print(f"评价用户数：{len(test_answers)}")
print(f"训练目录中的电影数：{len(movie_ids)}")
print(f"测试集冷启动记录：{cold_test_records}")
print(f"测试集冷启动电影：{len(cold_test_movies)}")
print()


# ============================================================
# 6. 定义头部电影与长尾电影
# ============================================================

number_of_head_movies = math.ceil(
    len(ranked_movies) * 0.20
)

head_movies = set(
    ranked_movies[:number_of_head_movies]
)


# ============================================================
# 7. 构造ItemCF相似度矩阵
# ============================================================

print("正在计算电影相似度矩阵……")

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

genre_intersection = (
    movie_genres @ movie_genres.T
)

genre_count = movie_genres.sum(axis=1)

genre_union = (
    genre_count[:, None]
    + genre_count[None, :]
    - genre_intersection
)

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
# 9. 计算推荐列表内部多样性
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
# 10. 多样性重排序
# ============================================================

def rerank_candidates(
    candidate_movies,
    raw_scores,
    normalized_scores,
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
                SELECTED_ALPHA
                * relevance_score
                + (1 - SELECTED_ALPHA)
                * diversity_score
            )

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
# 11. 为每位测试用户生成三种推荐
# ============================================================

print("正在生成测试集推荐……")

popularity_recommendations = {}
itemcf_recommendations = {}
reranked_recommendations = {}

for position, user_id in enumerate(
    test_answers,
    start=1,
):
    seen_movies = train_history.get(
        user_id,
        set(),
    )

    # --------------------------------------------------------
    # 热门推荐
    # --------------------------------------------------------

    popularity_list = [
        movie_id
        for movie_id in ranked_movies
        if movie_id not in seen_movies
    ][:TOP_K]

    popularity_recommendations[
        user_id
    ] = popularity_list

    # --------------------------------------------------------
    # ItemCF
    # --------------------------------------------------------

    seen_indices = [
        movie_to_index[movie_id]
        for movie_id in sorted(seen_movies)
        if movie_id in movie_to_index
    ]

    if seen_indices:
        scores = item_similarity[
            :,
            seen_indices,
        ].sum(axis=1)
    else:
        scores = np.zeros(
            len(movie_ids),
            dtype="float32",
        )

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
    )

    itemcf_recommendations[
        user_id
    ] = candidate_movies[:TOP_K]

    # --------------------------------------------------------
    # ItemCF + 多样性重排序
    # --------------------------------------------------------

    candidate_pool = candidate_movies[
        :CANDIDATE_POOL_SIZE
    ]

    raw_scores = np.array(
        [
            scores[
                movie_to_index[movie_id]
            ]
            for movie_id in candidate_pool
        ],
        dtype="float32",
    )

    minimum_score = raw_scores.min()
    maximum_score = raw_scores.max()

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

    reranked_recommendations[
        user_id
    ] = rerank_candidates(
        candidate_movies=candidate_pool,
        raw_scores=raw_scores,
        normalized_scores=normalized_scores,
    )

    if (
        position % 100 == 0
        or position == len(test_answers)
    ):
        print(
            f"已完成：{position}/"
            f"{len(test_answers)} 名用户"
        )


# ============================================================
# 12. 统一评价函数
# ============================================================

def evaluate_model(
    model_name,
    recommendations_by_user,
):
    user_results = []
    all_recommended_movies = []

    for user_id, relevant_movies in (
        test_answers.items()
    ):
        recommendations = (
            recommendations_by_user[
                user_id
            ]
        )

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

        reachable_relevant_count = len(
            relevant_movies & movie_id_set
        )

        user_results.append(
            {
                "user_id": user_id,
                "model": model_name,
                "relevant_count": len(
                    relevant_movies
                ),
                "reachable_relevant_count": (
                    reachable_relevant_count
                ),
                "hit_count": len(hits),
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

    average_popularity = float(
        np.mean(
            [
                item_popularity[movie_id]
                for movie_id
                in all_recommended_movies
            ]
        )
    )

    long_tail_share = float(
        np.mean(
            [
                movie_id not in head_movies
                for movie_id
                in all_recommended_movies
            ]
        )
    )

    summary = {
        "model": model_name,
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

    return summary, user_results


# ============================================================
# 13. 评价三种模型
# ============================================================

model_recommendations = {
    "popularity": (
        popularity_recommendations
    ),
    "itemcf_10": (
        itemcf_recommendations
    ),
    "itemcf_10_diversity_alpha_0.4": (
        reranked_recommendations
    ),
}

all_summaries = []
all_user_results = []

for model_name, recommendations in (
    model_recommendations.items()
):
    print(f"正在评价：{model_name}")

    summary, user_result = evaluate_model(
        model_name,
        recommendations,
    )

    all_summaries.append(summary)
    all_user_results.append(user_result)

final_summary = pd.DataFrame(
    all_summaries
)

final_per_user = pd.concat(
    all_user_results,
    ignore_index=True,
)


# ============================================================
# 14. 保存推荐清单
# ============================================================

recommendation_rows = []

for model_name, recommendations_by_user in (
    model_recommendations.items()
):
    for user_id, recommendations in (
        recommendations_by_user.items()
    ):
        for rank, movie_id in enumerate(
            recommendations,
            start=1,
        ):
            recommendation_rows.append(
                {
                    "model": model_name,
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
                    "is_test_relevant": (
                        movie_id
                        in test_answers[user_id]
                    ),
                }
            )

final_recommendations = pd.DataFrame(
    recommendation_rows
)


# ============================================================
# 15. 保存最终测试结果
# ============================================================

final_summary.to_csv(
    RESULTS_DIR
    / "final_test_summary.csv",
    index=False,
)

final_per_user.to_csv(
    RESULTS_DIR
    / "final_test_per_user.csv",
    index=False,
)

final_recommendations.to_csv(
    RESULTS_DIR
    / "final_test_recommendations.csv",
    index=False,
)


# ============================================================
# 16. 打印最终结果
# ============================================================

display_columns = [
    "model",
    "recall_at_10",
    "hit_rate_at_10",
    "ndcg_at_10",
    "intra_list_diversity",
    "catalog_coverage",
    "long_tail_share",
]

display_result = final_summary[
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
print("=" * 110)
print("最终测试集结果")
print("=" * 110)

print(
    display_result.to_string(
        index=False
    )
)

print()
print("锁定参数：")
print(
    f"ItemCF近邻数："
    f"{NEIGHBORS_PER_ITEM}"
)
print(
    f"候选池大小："
    f"{CANDIDATE_POOL_SIZE}"
)
print(
    f"多样性alpha："
    f"{SELECTED_ALPHA}"
)

print()
print("结果保存至：")
print(
    RESULTS_DIR
    / "final_test_summary.csv"
)
print(
    RESULTS_DIR
    / "final_test_per_user.csv"
)
print(
    RESULTS_DIR
    / "final_test_recommendations.csv"
)