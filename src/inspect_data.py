from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-100k"


ratings = pd.read_csv(
    DATA_DIR / "u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"],
)

movies = pd.read_csv(
    DATA_DIR / "u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"],
)


print("评分数据前5行：")
print(ratings.head())

print("\n电影数据前5行：")
print(movies.head())

print("\n数据规模：")
print(f"评分数量：{len(ratings)}")
print(f"用户数量：{ratings['user_id'].nunique()}")
print(f"电影数量：{ratings['movie_id'].nunique()}")

print("\n评分分布：")
print(ratings["rating"].value_counts().sort_index())

movie_statistics = (
    ratings.groupby("movie_id")
    .agg(
        rating_count=("rating", "size"),
        average_rating=("rating", "mean"),
    )
    .reset_index()
    .merge(movies, on="movie_id", how="left")
)

print("\n被评分次数最多的10部电影：")
print(
    movie_statistics
    .sort_values("rating_count", ascending=False)
    [["title", "rating_count", "average_rating"]]
    .head(10)
)