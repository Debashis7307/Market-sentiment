from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}. Found: {list(df.columns)}")


def load_trades(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(
        df,
        [
            "Account",
            "Coin",
            "Execution Price",
            "Size Tokens",
            "Size USD",
            "Side",
            "Timestamp IST",
            "Start Position",
            "Direction",
            "Closed PnL",
            "Fee",
        ],
        "historical_data.csv",
    )

    df = df.rename(
        columns={
            "Account": "account",
            "Coin": "symbol",
            "Execution Price": "execution_price",
            "Size Tokens": "size_tokens",
            "Size USD": "size_usd",
            "Side": "side",
            "Timestamp IST": "timestamp_ist",
            "Start Position": "start_position",
            "Direction": "direction",
            "Closed PnL": "closed_pnl",
            "Fee": "fee",
        }
    )

    df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"], format="%d-%m-%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["timestamp_ist"]).copy()
    df["date"] = df["timestamp_ist"].dt.normalize()

    for col in ["execution_price", "size_tokens", "size_usd", "start_position", "closed_pnl", "fee"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["side"] = df["side"].astype(str).str.upper().str.strip()
    df["direction"] = df["direction"].astype(str).str.strip()

    df["net_pnl"] = df["closed_pnl"].fillna(0.0) - df["fee"].fillna(0.0)
    df["is_close"] = df["closed_pnl"].fillna(0.0) != 0.0
    df["win"] = df["net_pnl"] > 0
    return df


def load_sentiment(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, ["date", "classification", "value"], "fear_greed_index.csv")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].dt.normalize()

    df["classification"] = df["classification"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    def normalize_bucket(x: str) -> str:
        x = x.lower().strip()
        if "fear" in x:
            return "Fear"
        if "greed" in x:
            return "Greed"
        return "Neutral"

    df["bucket"] = df["classification"].map(normalize_bucket)
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    return df[["date", "value", "classification", "bucket"]]


def savefig(out_dir: Path, filename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Market sentiment vs trader performance (Hyperliquid)")
    parser.add_argument("--trades", default="historical_data.csv", help="Path to historical trader data CSV")
    parser.add_argument("--sentiment", default="fear_greed_index.csv", help="Path to fear/greed index CSV")
    parser.add_argument("--out", default="outputs", help="Output directory for plots and summary CSVs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = load_trades(args.trades)
    sent = load_sentiment(args.sentiment)

    df = trades.merge(sent, on="date", how="inner")
    if df.empty:
        raise RuntimeError(
            "No overlapping dates between trades and sentiment. "
            "Check that trade timestamps and sentiment dates are in the same range."
        )

    sns.set_theme(style="whitegrid")

    coverage = df[["date"]].drop_duplicates().shape[0]
    print(f"Rows: {len(df):,}  Accounts: {df['account'].nunique():,}  Days matched: {coverage:,}")
    print("Sentiment bucket counts (rows):")
    print(df["bucket"].value_counts(dropna=False).to_string())

    daily = (
        df.groupby(["date", "bucket"], as_index=False)
        .agg(net_pnl=("net_pnl", "sum"), trades=("account", "size"))
        .sort_values("date")
    )

    trader = (
        df.groupby("account", as_index=False)
        .agg(
            trades=("account", "size"),
            close_trades=("is_close", "sum"),
            gross_pnl=("closed_pnl", "sum"),
            fees=("fee", "sum"),
            net_pnl=("net_pnl", "sum"),
            win_rate=("win", "mean"),
            avg_trade_usd=("size_usd", "mean"),
            median_trade_usd=("size_usd", "median"),
        )
        .sort_values("net_pnl", ascending=False)
    )
    trader.to_csv(out_dir / "trader_summary.csv", index=False)

    bucket_summary = (
        df.groupby("bucket", as_index=False)
        .agg(
            trades=("account", "size"),
            accounts=("account", "nunique"),
            net_pnl=("net_pnl", "sum"),
            avg_net_pnl=("net_pnl", "mean"),
            median_net_pnl=("net_pnl", "median"),
            win_rate=("win", "mean"),
            avg_trade_usd=("size_usd", "mean"),
        )
        .sort_values("net_pnl", ascending=False)
    )
    bucket_summary.to_csv(out_dir / "bucket_summary.csv", index=False)

    print("\nBucket summary (net):")
    print(bucket_summary.to_string(index=False))

    plt.figure(figsize=(7.5, 4.5))
    sns.barplot(data=bucket_summary, x="bucket", y="avg_net_pnl", hue="bucket", legend=False)
    plt.title("Average net PnL per trade by sentiment bucket")
    plt.xlabel("")
    plt.ylabel("Avg net PnL per trade")
    savefig(out_dir, "avg_net_pnl_by_bucket.png")

    plt.figure(figsize=(8.5, 4.5))
    sns.boxplot(data=df, x="bucket", y="net_pnl", showfliers=False)
    plt.axhline(0, color="black", linewidth=1, alpha=0.6)
    plt.title("Net PnL distribution by sentiment bucket (no outliers)")
    plt.xlabel("")
    plt.ylabel("Net PnL per trade")
    savefig(out_dir, "net_pnl_distribution_by_bucket.png")

    plt.figure(figsize=(10, 5))
    pivot = daily.pivot(index="date", columns="bucket", values="net_pnl").fillna(0.0)
    pivot.rolling(7, min_periods=1).mean().plot(ax=plt.gca())
    plt.axhline(0, color="black", linewidth=1, alpha=0.5)
    plt.title("7D rolling net PnL by sentiment bucket (daily aggregated)")
    plt.xlabel("")
    plt.ylabel("Net PnL (daily)")
    savefig(out_dir, "daily_net_pnl_7d_rolling_by_bucket.png")

    top_n = 25
    top = trader.head(top_n)["account"]
    top_df = df[df["account"].isin(top)].copy()

    mix = (
        top_df.groupby(["account", "bucket"], as_index=False)
        .agg(trades=("account", "size"), net_pnl=("net_pnl", "sum"))
        .sort_values(["account", "trades"], ascending=[True, False])
    )
    mix.to_csv(out_dir / "top_traders_bucket_mix.csv", index=False)

    plt.figure(figsize=(11, 6))
    heat = mix.pivot_table(index="account", columns="bucket", values="net_pnl", aggfunc="sum").fillna(0.0)
    sns.heatmap(heat, cmap="RdYlGn", center=0)
    plt.title(f"Top {top_n} accounts: net PnL by sentiment bucket")
    plt.xlabel("Sentiment bucket")
    plt.ylabel("Account")
    savefig(out_dir, "top_traders_net_pnl_heatmap.png")

    by_side = (
        df.groupby(["side", "bucket"], as_index=False)
        .agg(net_pnl=("net_pnl", "mean"), trades=("account", "size"))
        .sort_values("trades", ascending=False)
    )
    by_side.to_csv(out_dir / "side_bucket_summary.csv", index=False)

    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=by_side, x="bucket", y="net_pnl", hue="side")
    plt.title("Average net PnL per trade by side and sentiment bucket")
    plt.xlabel("")
    plt.ylabel("Avg net PnL per trade")
    savefig(out_dir, "avg_net_pnl_by_side_and_bucket.png")

    print(f"\nWrote outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()