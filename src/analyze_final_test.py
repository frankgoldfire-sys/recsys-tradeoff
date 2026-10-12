from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


# ============================================================
# 1. 路径与统计设置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

PER_USER_FILE = (
    RESULTS_DIR / "final_test_per_user.csv"
)

BOOTSTRAP_TIMES = 10000
RANDOM_SEED = 20261012
CONFIDENCE_LEVEL = 0.95

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# 2. 读取逐用户结果
# ============================================================

results = pd.read_csv(
    PER_USER_FILE
)

expected_models = {
    "popularity",
    "itemcf_10",
    "itemcf_10_diversity_alpha_0.4",
}

actual_models = set(
    results["model"].unique()
)

if actual_models != expected_models:
    raise ValueError(
        "模型名称或数量不符合预期。\n"
        f"预期：{expected_models}\n"
        f"实际：{actual_models}"
    )

duplicate_count = int(
    results.duplicated(
        ["user_id", "model"]
    ).sum()
)

if duplicate_count > 0:
    raise ValueError(
        f"发现重复的用户—模型记录："
        f"{duplicate_count}"
    )

model_counts = (
    results.groupby("model")["user_id"]
    .nunique()
)

if model_counts.nunique() != 1:
    raise ValueError(
        "三个模型的评价用户数量不一致。"
    )

print("数据完整性检查通过")
print(f"总记录数：{len(results)}")
print(f"每个模型用户数：{model_counts.iloc[0]}")
print()


# ============================================================
# 3. 配对Bootstrap置信区间
# ============================================================

def paired_bootstrap(
    reference_values,
    challenger_values,
):
    differences = (
        challenger_values
        - reference_values
    )

    user_count = len(differences)

    bootstrap_means = np.empty(
        BOOTSTRAP_TIMES,
        dtype="float64",
    )

    batch_size = 1000

    for start in range(
        0,
        BOOTSTRAP_TIMES,
        batch_size,
    ):
        end = min(
            start + batch_size,
            BOOTSTRAP_TIMES,
        )

        current_batch_size = (
            end - start
        )

        sampled_indices = rng.integers(
            low=0,
            high=user_count,
            size=(
                current_batch_size,
                user_count,
            ),
        )

        bootstrap_means[start:end] = (
            differences[sampled_indices]
            .mean(axis=1)
        )

    alpha = (
        1 - CONFIDENCE_LEVEL
    ) / 2

    lower_bound = np.quantile(
        bootstrap_means,
        alpha,
    )

    upper_bound = np.quantile(
        bootstrap_means,
        1 - alpha,
    )

    return {
        "reference_mean": (
            reference_values.mean()
        ),
        "challenger_mean": (
            challenger_values.mean()
        ),
        "mean_difference": (
            differences.mean()
        ),
        "ci_lower": lower_bound,
        "ci_upper": upper_bound,
    }


# ============================================================
# 4. 预先规定的两组核心比较
# ============================================================

comparisons = [
    {
        "comparison": (
            "popularity_to_itemcf"
        ),
        "reference": "popularity",
        "challenger": "itemcf_10",
    },
    {
        "comparison": (
            "itemcf_to_diversity_reranking"
        ),
        "reference": "itemcf_10",
        "challenger": (
            "itemcf_10_diversity_alpha_0.4"
        ),
    },
]

metrics = [
    "recall_at_10",
    "hit_rate_at_10",
    "ndcg_at_10",
    "intra_list_diversity",
]

bootstrap_rows = []

for comparison in comparisons:
    reference_name = comparison[
        "reference"
    ]

    challenger_name = comparison[
        "challenger"
    ]

    for metric in metrics:
        metric_table = results.pivot(
            index="user_id",
            columns="model",
            values=metric,
        ).dropna(
            subset=[
                reference_name,
                challenger_name,
            ]
        )

        reference_values = metric_table[
            reference_name
        ].to_numpy(dtype="float64")

        challenger_values = metric_table[
            challenger_name
        ].to_numpy(dtype="float64")

        statistics = paired_bootstrap(
            reference_values,
            challenger_values,
        )

        bootstrap_rows.append(
            {
                "comparison": comparison[
                    "comparison"
                ],
                "reference": reference_name,
                "challenger": challenger_name,
                "metric": metric,
                "user_count": len(
                    metric_table
                ),
                **statistics,
            }
        )

bootstrap_results = pd.DataFrame(
    bootstrap_rows
)


# ============================================================
# 5. Hit Rate的McNemar精确检验
# ============================================================

mcnemar_rows = []

for comparison in comparisons:
    reference_name = comparison[
        "reference"
    ]

    challenger_name = comparison[
        "challenger"
    ]

    hit_table = results.pivot(
        index="user_id",
        columns="model",
        values="hit_rate_at_10",
    ).dropna(
        subset=[
            reference_name,
            challenger_name,
        ]
    )

    reference_hits = hit_table[
        reference_name
    ].to_numpy(dtype=int)

    challenger_hits = hit_table[
        challenger_name
    ].to_numpy(dtype=int)

    reference_only = int(
        np.sum(
            (reference_hits == 1)
            & (challenger_hits == 0)
        )
    )

    challenger_only = int(
        np.sum(
            (reference_hits == 0)
            & (challenger_hits == 1)
        )
    )

    discordant_count = (
        reference_only
        + challenger_only
    )

    if discordant_count == 0:
        p_value = 1.0
    else:
        p_value = binomtest(
            challenger_only,
            n=discordant_count,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    mcnemar_rows.append(
        {
            "comparison": comparison[
                "comparison"
            ],
            "reference": reference_name,
            "challenger": challenger_name,
            "reference_only_hit": (
                reference_only
            ),
            "challenger_only_hit": (
                challenger_only
            ),
            "hit_rate_difference": (
                challenger_hits.mean()
                - reference_hits.mean()
            ),
            "exact_p_value": p_value,
        }
    )

mcnemar_results = pd.DataFrame(
    mcnemar_rows
)


# ============================================================
# 6. 保存结果
# ============================================================

bootstrap_results.to_csv(
    RESULTS_DIR
    / "final_test_paired_bootstrap.csv",
    index=False,
)

mcnemar_results.to_csv(
    RESULTS_DIR
    / "final_test_mcnemar.csv",
    index=False,
)


# ============================================================
# 7. 打印Bootstrap结果
# ============================================================

metric_names = {
    "recall_at_10": "Recall@10",
    "hit_rate_at_10": "Hit Rate@10",
    "ndcg_at_10": "NDCG@10",
    "intra_list_diversity": (
        "列表内部多样性"
    ),
}

print("=" * 80)
print("配对Bootstrap分析")
print("=" * 80)

for comparison in comparisons:
    comparison_name = comparison[
        "comparison"
    ]

    print()
    print(comparison_name)

    current_rows = bootstrap_results[
        bootstrap_results["comparison"]
        == comparison_name
    ]

    for _, row in current_rows.iterrows():
        metric_name = metric_names[
            row["metric"]
        ]

        print(
            f"  {metric_name}"
        )
        print(
            "    参照模型："
            f"{row['reference_mean']:.2%}"
        )
        print(
            "    对比模型："
            f"{row['challenger_mean']:.2%}"
        )
        print(
            "    平均差值："
            f"{row['mean_difference']:+.2%}"
        )
        print(
            "    95%置信区间："
            f"[{row['ci_lower']:+.2%}, "
            f"{row['ci_upper']:+.2%}]"
        )


# ============================================================
# 8. 打印McNemar结果
# ============================================================

print()
print("=" * 80)
print("Hit Rate@10的McNemar精确检验")
print("=" * 80)

for _, row in mcnemar_results.iterrows():
    print()
    print(row["comparison"])

    print(
        "  只有参照模型命中："
        f"{int(row['reference_only_hit'])}"
    )

    print(
        "  只有对比模型命中："
        f"{int(row['challenger_only_hit'])}"
    )

    print(
        "  Hit Rate差值："
        f"{row['hit_rate_difference']:+.2%}"
    )

    print(
        "  精确检验p值："
        f"{row['exact_p_value']:.4f}"
    )


print()
print("统计分析结果已保存至：")
print(
    RESULTS_DIR
    / "final_test_paired_bootstrap.csv"
)
print(
    RESULTS_DIR
    / "final_test_mcnemar.csv"
)