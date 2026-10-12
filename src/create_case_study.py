from pathlib import Path

import pandas as pd


# ============================================================
# 1. 路径和模型名称
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

RAW_DATA_DIR = (
    PROJECT_ROOT / "data" / "raw" / "ml-100k"
)

RESULTS_DIR = PROJECT_ROOT / "results"

REFERENCE_MODEL = "itemcf_10"

RERANKED_MODEL = (
    "itemcf_10_diversity_alpha_0.4"
)


# ============================================================
# 2. 读取最终结果
# ============================================================

per_user = pd.read_csv(
    RESULTS_DIR / "final_test_per_user.csv"
)

recommendations = pd.read_csv(
    RESULTS_DIR
    / "final_test_recommendations.csv"
)

train = pd.read_csv(
    PROCESSED_DATA_DIR / "train.csv"
)

validation = pd.read_csv(
    PROCESSED_DATA_DIR / "validation.csv"
)

final_train = pd.concat(
    [train, validation],
    ignore_index=True,
)


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

movie_table = movies.set_index(
    "movie_id"
)


# ============================================================
# 4. 将电影类型转成可阅读文字
# ============================================================

def get_genre_text(movie_id):
    if movie_id not in movie_table.index:
        return "Unknown"

    movie_row = movie_table.loc[
        movie_id
    ]

    genres = [
        genre
        for genre in genre_columns
        if movie_row[genre] == 1
    ]

    if not genres:
        return "Unknown"

    return ", ".join(genres)


# ============================================================
# 5. 整理两种模型的逐用户结果
# ============================================================

reference_results = (
    per_user[
        per_user["model"]
        == REFERENCE_MODEL
    ]
    .set_index("user_id")
)

reranked_results = (
    per_user[
        per_user["model"]
        == RERANKED_MODEL
    ]
    .set_index("user_id")
)

comparison = (
    reference_results[
        [
            "hit_count",
            "recall_at_10",
            "hit_rate_at_10",
            "ndcg_at_10",
            "intra_list_diversity",
        ]
    ]
    .join(
        reranked_results[
            [
                "hit_count",
                "recall_at_10",
                "hit_rate_at_10",
                "ndcg_at_10",
                "intra_list_diversity",
            ]
        ],
        lsuffix="_itemcf",
        rsuffix="_reranked",
        how="inner",
    )
)

comparison["diversity_gain"] = (
    comparison[
        "intra_list_diversity_reranked"
    ]
    - comparison[
        "intra_list_diversity_itemcf"
    ]
)

comparison["ndcg_change"] = (
    comparison["ndcg_at_10_reranked"]
    - comparison["ndcg_at_10_itemcf"]
)


# ============================================================
# 6. 按预定规则选择代表性用户
# ============================================================

median_diversity_gain = comparison[
    "diversity_gain"
].median()

# 优先选择两个模型都至少命中一部，
# 并且命中数量相同的用户
same_hit_candidates = comparison[
    (
        comparison["hit_count_itemcf"]
        == comparison["hit_count_reranked"]
    )
    & (
        comparison["hit_count_itemcf"] > 0
    )
].copy()

# 如果找不到，再退回“命中数量相同”的条件
if same_hit_candidates.empty:
    same_hit_candidates = comparison[
        comparison["hit_count_itemcf"]
        == comparison["hit_count_reranked"]
    ].copy()

# 如果仍然找不到，才使用全部用户
if same_hit_candidates.empty:
    candidate_users = comparison.copy()
else:
    candidate_users = (
        same_hit_candidates.copy()
    )

candidate_users[
    "distance_to_median"
] = (
    candidate_users["diversity_gain"]
    - median_diversity_gain
).abs()

selected_user_id = int(
    candidate_users
    .reset_index()
    .sort_values(
        [
            "distance_to_median",
            "user_id",
        ],
        ascending=[
            True,
            True,
        ],
        kind="mergesort",
    )
    .iloc[0]["user_id"]
)

selected_metrics = comparison.loc[
    selected_user_id
]


# ============================================================
# 7. 整理用户最近的正向历史
# ============================================================

user_history = (
    final_train[
        final_train["user_id"]
        == selected_user_id
    ]
    .sort_values(
        "timestamp",
        kind="mergesort",
    )
    .tail(10)
    .copy()
)

user_history["title"] = (
    user_history["movie_id"]
    .map(movie_table["title"])
)

user_history["genres"] = (
    user_history["movie_id"]
    .map(get_genre_text)
)

history_output = user_history[
    [
        "user_id",
        "movie_id",
        "title",
        "genres",
        "rating",
        "timestamp",
    ]
].copy()


# ============================================================
# 8. 整理两种推荐列表
# ============================================================

case_recommendations = recommendations[
    (
        recommendations["user_id"]
        == selected_user_id
    )
    & (
        recommendations["model"]
        .isin(
            [
                REFERENCE_MODEL,
                RERANKED_MODEL,
            ]
        )
    )
].copy()

case_recommendations["genres"] = (
    case_recommendations["movie_id"]
    .map(get_genre_text)
)

itemcf_movies = set(
    case_recommendations.loc[
        case_recommendations["model"]
        == REFERENCE_MODEL,
        "movie_id",
    ]
)

reranked_movies = set(
    case_recommendations.loc[
        case_recommendations["model"]
        == RERANKED_MODEL,
        "movie_id",
    ]
)

case_recommendations[
    "retained_in_both_lists"
] = (
    case_recommendations["movie_id"]
    .isin(
        itemcf_movies
        & reranked_movies
    )
)

case_recommendations = (
    case_recommendations.sort_values(
        ["model", "rank"],
        kind="mergesort",
    )
)


# ============================================================
# 9. 保存案例结果
# ============================================================

case_summary = pd.DataFrame(
    [
        {
            "selected_user_id": (
                selected_user_id
            ),
            "selection_rule": (
             "两个模型均有正向命中、命中数量相同，"
             "且多样性提升最接近全体中位数"
),
            "overall_median_diversity_gain": (
                median_diversity_gain
            ),
            "itemcf_hit_count": (
                selected_metrics[
                    "hit_count_itemcf"
                ]
            ),
            "reranked_hit_count": (
                selected_metrics[
                    "hit_count_reranked"
                ]
            ),
            "itemcf_ndcg_at_10": (
                selected_metrics[
                    "ndcg_at_10_itemcf"
                ]
            ),
            "reranked_ndcg_at_10": (
                selected_metrics[
                    "ndcg_at_10_reranked"
                ]
            ),
            "itemcf_diversity": (
                selected_metrics[
                    "intra_list_diversity_itemcf"
                ]
            ),
            "reranked_diversity": (
                selected_metrics[
                    "intra_list_diversity_reranked"
                ]
            ),
            "diversity_gain": (
                selected_metrics[
                    "diversity_gain"
                ]
            ),
        }
    ]
)

case_summary.to_csv(
    RESULTS_DIR
    / "representative_case_summary.csv",
    index=False,
)

history_output.to_csv(
    RESULTS_DIR
    / "representative_case_history.csv",
    index=False,
)

case_recommendations.to_csv(
    RESULTS_DIR
    / "representative_case_recommendations.csv",
    index=False,
)


# ============================================================
# 10. 打印案例
# ============================================================

print("=" * 80)
print("代表性用户案例")
print("=" * 80)

print(
    f"选中的用户ID：{selected_user_id}"
)

print(
    "全体用户多样性提升中位数："
    f"{median_diversity_gain:+.2%}"
)

print()
print("该用户的模型表现：")

print(
    "ItemCF："
    f"命中{int(selected_metrics['hit_count_itemcf'])}部，"
    f"NDCG={selected_metrics['ndcg_at_10_itemcf']:.2%}，"
    "多样性="
    f"{selected_metrics['intra_list_diversity_itemcf']:.2%}"
)

print(
    "重排序："
    f"命中{int(selected_metrics['hit_count_reranked'])}部，"
    f"NDCG={selected_metrics['ndcg_at_10_reranked']:.2%}，"
    "多样性="
    f"{selected_metrics['intra_list_diversity_reranked']:.2%}"
)

print(
    "多样性变化："
    f"{selected_metrics['diversity_gain']:+.2%}"
)

print()
print("-" * 90)
print("该用户最近10条正向历史")
print("-" * 90)

print(
    history_output[
        [
            "movie_id",
            "title",
            "genres",
        ]
    ].to_string(
        index=False
    )
)

for model_name, display_name in [
    (
        REFERENCE_MODEL,
        "原始ItemCF推荐",
    ),
    (
        RERANKED_MODEL,
        "多样性重排序推荐",
    ),
]:
    print()
    print("-" * 90)
    print(display_name)
    print("-" * 90)

    current_list = case_recommendations[
        case_recommendations["model"]
        == model_name
    ]

    print(
        current_list[
            [
                "rank",
                "movie_id",
                "title",
                "genres",
                "is_test_relevant",
                "is_long_tail",
            ]
        ].to_string(
            index=False
        )
    )

print()
print("案例结果已保存至：")

print(
    RESULTS_DIR
    / "representative_case_summary.csv"
)

print(
    RESULTS_DIR
    / "representative_case_history.csv"
)

print(
    RESULTS_DIR
    / "representative_case_recommendations.csv"
)