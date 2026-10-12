from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


# ============================================================
# 1. 路径与绘图设置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL_SUMMARY_FILE = (
    RESULTS_DIR / "final_test_summary.csv"
)

TUNING_FILE = (
    RESULTS_DIR
    / "diversity_reranking_tuning_summary.csv"
)

BOOTSTRAP_FILE = (
    RESULTS_DIR
    / "final_test_paired_bootstrap.csv"
)

SELECTED_ALPHA = 0.4


# ============================================================
# 2. 设置中文字体
# ============================================================

available_fonts = {
    font.name
    for font in font_manager.fontManager.ttflist
}

font_candidates = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]

selected_font = None

for font_name in font_candidates:
    if font_name in available_fonts:
        selected_font = font_name
        break

if selected_font is not None:
    plt.rcParams["font.sans-serif"] = [
        selected_font
    ]

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300


# ============================================================
# 3. 读取结果
# ============================================================

final_summary = pd.read_csv(
    FINAL_SUMMARY_FILE
)

tuning_summary = pd.read_csv(
    TUNING_FILE
)

bootstrap_results = pd.read_csv(
    BOOTSTRAP_FILE
)


# ============================================================
# 4. 模型名称与颜色
# ============================================================

model_order = [
    "popularity",
    "itemcf_10",
    "itemcf_10_diversity_alpha_0.4",
]

model_labels = {
    "popularity": "热门推荐",
    "itemcf_10": "ItemCF",
    "itemcf_10_diversity_alpha_0.4": (
        "ItemCF + 多样性重排序"
    ),
}

model_colors = {
    "popularity": "#8C8C8C",
    "itemcf_10": "#4C78A8",
    "itemcf_10_diversity_alpha_0.4": (
        "#F58518"
    ),
}

missing_models = [
    model
    for model in model_order
    if model
    not in set(final_summary["model"])
]

if missing_models:
    raise ValueError(
        "最终结果中缺少模型："
        f"{missing_models}"
    )

final_summary = (
    final_summary
    .set_index("model")
    .reindex(model_order)
)


# ============================================================
# 5. 图一：最终测试集模型比较
# ============================================================

accuracy_metrics = [
    "recall_at_10",
    "hit_rate_at_10",
    "ndcg_at_10",
]

accuracy_labels = [
    "Recall@10",
    "Hit Rate@10",
    "NDCG@10",
]

distribution_metrics = [
    "intra_list_diversity",
    "catalog_coverage",
    "long_tail_share",
]

distribution_labels = [
    "列表内部多样性",
    "目录覆盖率",
    "长尾电影占比",
]

figure, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(14, 6),
)

bar_width = 0.24

# ------------------------------------------------------------
# 左图：准确性指标
# ------------------------------------------------------------

x_accuracy = np.arange(
    len(accuracy_metrics)
)

for model_index, model_name in enumerate(
    model_order
):
    values = (
        final_summary.loc[
            model_name,
            accuracy_metrics,
        ].to_numpy(dtype=float)
        * 100
    )

    positions = (
        x_accuracy
        + (
            model_index
            - (len(model_order) - 1) / 2
        )
        * bar_width
    )

    bars = axes[0].bar(
        positions,
        values,
        width=bar_width,
        color=model_colors[model_name],
        label=model_labels[model_name],
    )

    axes[0].bar_label(
        bars,
        labels=[
            f"{value:.1f}%"
            for value in values
        ],
        padding=3,
        fontsize=9,
    )

axes[0].set_title(
    "测试集准确性指标"
)

axes[0].set_ylabel(
    "比例（%）"
)

axes[0].set_xticks(
    x_accuracy,
    accuracy_labels,
)

accuracy_maximum = (
    final_summary[
        accuracy_metrics
    ].to_numpy(dtype=float).max()
    * 100
)

axes[0].set_ylim(
    0,
    accuracy_maximum * 1.28,
)

axes[0].grid(
    axis="y",
    linestyle="--",
    alpha=0.25,
)

# ------------------------------------------------------------
# 右图：分布与多样性指标
# ------------------------------------------------------------

x_distribution = np.arange(
    len(distribution_metrics)
)

for model_index, model_name in enumerate(
    model_order
):
    values = (
        final_summary.loc[
            model_name,
            distribution_metrics,
        ].to_numpy(dtype=float)
        * 100
    )

    positions = (
        x_distribution
        + (
            model_index
            - (len(model_order) - 1) / 2
        )
        * bar_width
    )

    bars = axes[1].bar(
        positions,
        values,
        width=bar_width,
        color=model_colors[model_name],
        label=model_labels[model_name],
    )

    axes[1].bar_label(
        bars,
        labels=[
            f"{value:.1f}%"
            for value in values
        ],
        padding=3,
        fontsize=9,
    )

axes[1].set_title(
    "测试集分布与多样性指标"
)

axes[1].set_ylabel(
    "比例（%）"
)

axes[1].set_xticks(
    x_distribution,
    distribution_labels,
)

axes[1].set_ylim(
    0,
    100,
)

axes[1].grid(
    axis="y",
    linestyle="--",
    alpha=0.25,
)

handles, labels = axes[0].get_legend_handles_labels()

figure.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
)

figure.suptitle(
    "三种推荐方法的最终测试集表现",
    fontsize=16,
    y=1.09,
)

figure.tight_layout()

figure.savefig(
    FIGURES_DIR
    / "final_test_model_comparison.png",
    bbox_inches="tight",
)

plt.close(figure)


# ============================================================
# 6. 图二：验证集准确性—多样性权衡曲线
# ============================================================

tradeoff_data = tuning_summary.sort_values(
    "intra_list_diversity"
).copy()

tradeoff_data[
    "ndcg_percent"
] = (
    tradeoff_data["ndcg_at_10"]
    * 100
)

tradeoff_data[
    "diversity_percent"
] = (
    tradeoff_data[
        "intra_list_diversity"
    ]
    * 100
)

base_ndcg = float(
    tuning_summary.loc[
        np.isclose(
            tuning_summary["alpha"],
            1.0,
        ),
        "ndcg_at_10",
    ].iloc[0]
)

minimum_allowed_ndcg = (
    base_ndcg * 0.95
)

figure, axis = plt.subplots(
    figsize=(9, 6.5)
)

axis.plot(
    tradeoff_data[
        "diversity_percent"
    ],
    tradeoff_data[
        "ndcg_percent"
    ],
    color="#6B7280",
    linewidth=1.8,
    zorder=1,
)

axis.scatter(
    tradeoff_data[
        "diversity_percent"
    ],
    tradeoff_data[
        "ndcg_percent"
    ],
    color="#4C78A8",
    s=60,
    zorder=2,
    label="候选参数",
)

selected_row = tuning_summary.loc[
    np.isclose(
        tuning_summary["alpha"],
        SELECTED_ALPHA,
    )
].iloc[0]

axis.scatter(
    selected_row[
        "intra_list_diversity"
    ] * 100,
    selected_row[
        "ndcg_at_10"
    ] * 100,
    color="#F58518",
    edgecolor="black",
    linewidth=0.8,
    s=150,
    zorder=4,
    label="最终选择：α=0.4",
)

axis.axhline(
    minimum_allowed_ndcg * 100,
    color="#C44E52",
    linestyle="--",
    linewidth=1.5,
    label=(
        "最低允许NDCG@10："
        f"{minimum_allowed_ndcg:.2%}"
    ),
)

for _, row in tradeoff_data.iterrows():
    axis.annotate(
        f"α={row['alpha']:.1f}",
        (
            row["diversity_percent"],
            row["ndcg_percent"],
        ),
        xytext=(5, 6),
        textcoords="offset points",
        fontsize=9,
    )

axis.set_title(
    "验证集上的准确性—多样性权衡"
)

axis.set_xlabel(
    "列表内部多样性（%）"
)

axis.set_ylabel(
    "NDCG@10（%）"
)

axis.grid(
    linestyle="--",
    alpha=0.25,
)

axis.legend(
    frameon=False,
    loc="lower left",
)

figure.tight_layout()

figure.savefig(
    FIGURES_DIR
    / "validation_accuracy_diversity_tradeoff.png",
    bbox_inches="tight",
)

plt.close(figure)


# ============================================================
# 7. 图三：重排序相对ItemCF的配对差值
# ============================================================

comparison_name = (
    "itemcf_to_diversity_reranking"
)

ci_data = bootstrap_results[
    bootstrap_results["comparison"]
    == comparison_name
].copy()

metric_order = [
    "recall_at_10",
    "hit_rate_at_10",
    "ndcg_at_10",
    "intra_list_diversity",
]

metric_labels = {
    "recall_at_10": "Recall@10",
    "hit_rate_at_10": "Hit Rate@10",
    "ndcg_at_10": "NDCG@10",
    "intra_list_diversity": (
        "列表内部多样性"
    ),
}

ci_data = (
    ci_data
    .set_index("metric")
    .reindex(metric_order)
)

if ci_data["mean_difference"].isna().any():
    raise ValueError(
        "Bootstrap结果中缺少需要绘制的指标。"
    )

mean_differences = (
    ci_data["mean_difference"]
    .to_numpy(dtype=float)
    * 100
)

lower_bounds = (
    ci_data["ci_lower"]
    .to_numpy(dtype=float)
    * 100
)

upper_bounds = (
    ci_data["ci_upper"]
    .to_numpy(dtype=float)
    * 100
)

lower_errors = (
    mean_differences
    - lower_bounds
)

upper_errors = (
    upper_bounds
    - mean_differences
)

y_positions = np.arange(
    len(metric_order)
)

figure, axis = plt.subplots(
    figsize=(9, 5.5)
)

for position, metric_name in enumerate(
    metric_order
):
    color = (
        "#F58518"
        if metric_name
        == "intra_list_diversity"
        else "#4C78A8"
    )

    axis.errorbar(
        mean_differences[position],
        position,
        xerr=np.array(
            [
                [
                    lower_errors[position]
                ],
                [
                    upper_errors[position]
                ],
            ]
        ),
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=2,
        capsize=5,
        markersize=8,
        zorder=3,
    )

    axis.text(
        upper_bounds[position] + 0.25,
        position,
        (
            f"{mean_differences[position]:+.2f} "
            "个百分点"
        ),
        va="center",
        fontsize=10,
    )

axis.axvline(
    0,
    color="#444444",
    linestyle="--",
    linewidth=1.2,
)

axis.set_yticks(
    y_positions,
    [
        metric_labels[metric]
        for metric in metric_order
    ],
)

axis.invert_yaxis()

axis.set_xlabel(
    "重排序相对ItemCF的平均差值（百分点）"
)

axis.set_title(
    "重排序效果及95% Bootstrap置信区间"
)

axis.grid(
    axis="x",
    linestyle="--",
    alpha=0.25,
)

minimum_x = min(
    lower_bounds.min(),
    0,
)

maximum_x = max(
    upper_bounds.max(),
    0,
)

x_padding = (
    maximum_x - minimum_x
) * 0.18

axis.set_xlim(
    minimum_x - x_padding,
    maximum_x + x_padding,
)

figure.tight_layout()

figure.savefig(
    FIGURES_DIR
    / "reranking_bootstrap_differences.png",
    bbox_inches="tight",
)

plt.close(figure)


# ============================================================
# 8. 完成
# ============================================================

print("最终图表绘制完成")
print("=" * 70)

print(
    FIGURES_DIR
    / "final_test_model_comparison.png"
)

print(
    FIGURES_DIR
    / "validation_accuracy_diversity_tradeoff.png"
)

print(
    FIGURES_DIR
    / "reranking_bootstrap_differences.png"
)