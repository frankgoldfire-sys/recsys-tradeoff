from pathlib import Path

import pandas as pd


# ============================================================
# 1. 确定文件位置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ml-100k" / "u.data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 读取数据并定义“喜欢”
# ============================================================

ratings = pd.read_csv(
    RAW_DATA_PATH,
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"],
)

# 预先规定：4分和5分表示用户喜欢这部电影
positive_ratings = ratings[ratings["rating"] >= 4].copy()

# 为了保证每位用户都有足够的数据用于训练、验证和测试，
# 只保留至少拥有10条正向记录的用户
positive_counts = positive_ratings.groupby("user_id").size()

eligible_users = positive_counts[
    positive_counts >= 10
].index

positive_ratings = positive_ratings[
    positive_ratings["user_id"].isin(eligible_users)
].copy()

# 先按用户排列，再按评分发生时间排列
positive_ratings = positive_ratings.sort_values(
    ["user_id", "timestamp", "movie_id"]
)


# ============================================================
# 3. 按照时间为每位用户切分数据
# ============================================================

train_parts = []
validation_parts = []
test_parts = []

for user_id, user_data in positive_ratings.groupby("user_id"):
    user_data = user_data.sort_values(
        ["timestamp", "movie_id"]
    )

    number_of_records = len(user_data)

    number_of_test = max(
        1,
        int(number_of_records * 0.10),
    )

    number_of_validation = max(
        1,
        int(number_of_records * 0.10),
    )

    train_end = (
        number_of_records
        - number_of_validation
        - number_of_test
    )

    validation_end = (
        number_of_records
        - number_of_test
    )

    train_parts.append(
        user_data.iloc[:train_end]
    )

    validation_parts.append(
        user_data.iloc[train_end:validation_end]
    )

    test_parts.append(
        user_data.iloc[validation_end:]
    )


train = pd.concat(
    train_parts,
    ignore_index=True,
)

validation = pd.concat(
    validation_parts,
    ignore_index=True,
)

test = pd.concat(
    test_parts,
    ignore_index=True,
)


# ============================================================
# 4. 处理训练集中从未出现过的电影
# ============================================================

train_movie_ids = set(train["movie_id"])

cold_validation = validation[
    ~validation["movie_id"].isin(train_movie_ids)
].copy()

cold_test = test[
    ~test["movie_id"].isin(train_movie_ids)
].copy()

validation = validation[
    validation["movie_id"].isin(train_movie_ids)
].copy()

test = test[
    test["movie_id"].isin(train_movie_ids)
].copy()


# ============================================================
# 5. 保存切分结果
# ============================================================

train.to_csv(
    PROCESSED_DATA_DIR / "train.csv",
    index=False,
)

validation.to_csv(
    PROCESSED_DATA_DIR / "validation.csv",
    index=False,
)

test.to_csv(
    PROCESSED_DATA_DIR / "test.csv",
    index=False,
)


# ============================================================
# 6. 输出检查结果
# ============================================================

print("数据切分完成")
print("=" * 50)

print(f"原始评分数量：{len(ratings)}")
print(f"4分及以上的评分数量：{len(ratings[ratings['rating'] >= 4])}")
print(f"符合条件的用户数量：{len(eligible_users)}")
print(f"用于切分的正向记录：{len(positive_ratings)}")

print()
print(f"训练集记录数：{len(train)}")
print(f"验证集记录数：{len(validation)}")
print(f"测试集记录数：{len(test)}")

print()
print(f"验证集冷启动记录：{len(cold_validation)}")
print(f"测试集冷启动记录：{len(cold_test)}")

print()
print("保存位置：")
print(PROCESSED_DATA_DIR)