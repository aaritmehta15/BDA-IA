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


