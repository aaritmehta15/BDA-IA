import os
import sys
import time

# Output directory for saved visualisations (Section 8.5)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Output")

# ─── Path constants ───────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_DIR  = os.path.join(PROJECT_ROOT, "Dataset")
FAKE_CSV     = os.path.join(DATASET_DIR, "Fake.csv")
TRUE_CSV     = os.path.join(DATASET_DIR, "True.csv")

USE_SPARK = False   # Will be flipped to True if Spark init succeeds


# ─── 1. Spark initialisation ───────────────────────────────────────────
def init_spark():
    global USE_SPARK

    # Force network binding so Spark doesn't hang on Windows
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    try:
        import findspark
        findspark.init()
    except Exception:
        pass  # findspark is optional if SPARK_HOME is already set

    try:
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder
            .appName("FakeNewsDetection")
            .master("local[*]")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.memory", "4g")
            .config(
                "spark.driver.extraJavaOptions",
                " ".join([
                    "-XX:+IgnoreUnrecognizedVMOptions",
                    "--add-opens=java.base/java.lang=ALL-UNNAMED",
                    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
                    "--add-opens=java.base/java.io=ALL-UNNAMED",
                    "--add-opens=java.base/java.net=ALL-UNNAMED",
                    "--add-opens=java.base/java.nio=ALL-UNNAMED",
                    "--add-opens=java.base/java.util=ALL-UNNAMED",
                    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
                    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
                    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
                    "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
                    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
                ])
            )
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        USE_SPARK = True
        print(f"[OK]  Spark {spark.version} initialised (local mode)")
        return spark

    except Exception as e:
        print(f"[WARN] Spark init failed: {e}")
        print("[INFO] Falling back to pandas mode — output structure is identical")
        return None


# ─── 2. Dataset loading ────────────────────────────────────────────────────────
def load_data_spark(spark):
    from pyspark.sql.functions import lit

    fake_df = spark.read.csv(FAKE_CSV, header=True, inferSchema=True).withColumn("label", lit(1))
    true_df = spark.read.csv(TRUE_CSV, header=True, inferSchema=True).withColumn("label", lit(0))
    return fake_df.union(true_df)


def load_data_pandas():
    import pandas as pd

    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)
    fake_df["label"] = 1
    true_df["label"] = 0
    return pd.concat([fake_df, true_df], ignore_index=True)


def load_data(spark=None):
    for path in (FAKE_CSV, TRUE_CSV):
        if not os.path.exists(path):
            sys.exit(f"[ERROR] Dataset not found: {path}\n"
                     "        Download from https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset "
                     "and place Fake.csv / True.csv in Group_XX_Implementation/Dataset/")

    if USE_SPARK and spark is not None:
        df = load_data_spark(spark)
        print("[OK]  Datasets loaded with PySpark")
    else:
        df = load_data_pandas()
        print("[OK]  Datasets loaded with pandas")

    return df


# ─── 3. Display helpers ─────────────────────────────────────────
def show_info(df):
    if USE_SPARK:
        print("\n── Schema ──────────────────────────────────────")
        df.printSchema()
        print("\n── First 5 rows ────────────────────────────────")
        df.show(5, truncate=80)
        print(f"\n── Total rows: {df.count():,} ─────────────────────────")
    else:
        print("\n── Columns / dtypes ────────────────────────────")
        print(df.dtypes)
        print("\n── First 5 rows ────────────────────────────────")
        print(df.head(5).to_string(index=False))
        print(f"\n── Total rows: {len(df):,} ─────────────────────────")


def show_dataset_stats(df, feat_dim=None):
    """Print dataset composition stats required by draft Section 8.1."""
    print("\n── Dataset Composition (Section 8.1) ────────────────")
    if USE_SPARK:
        total     = df.count()
        fake_cnt  = df.filter(df.label == 1).count()
        true_cnt  = df.filter(df.label == 0).count()
    else:
        total     = len(df)
        fake_cnt  = int((df["label"] == 1).sum())
        true_cnt  = int((df["label"] == 0).sum())
    print(f"  Total Labeled Records : {total:,}")
    print(f"  Fake News (label=1)   : {fake_cnt:,}")
    print(f"  True News (label=0)   : {true_cnt:,}")
    print(f"  Train / Test Split    : 80% / 20%")
    if feat_dim is not None:
        print(f"  TF-IDF Feature Dim    : {feat_dim:,}")
    return total, fake_cnt, true_cnt


