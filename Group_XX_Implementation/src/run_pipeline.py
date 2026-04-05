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

# ─── 4. Preprocessing ─────────────────────────────────────────────────────────

# English stopwords — used in pandas path
_STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and",
    "any","are","aren't","as","at","be","because","been","before","being",
    "below","between","both","but","by","can't","cannot","could","couldn't",
    "did","didn't","do","does","doesn't","doing","don't","down","during",
    "each","few","for","from","further","get","got","had","hadn't","has",
    "hasn't","have","haven't","having","he","he'd","he'll","he's","her",
    "here","here's","hers","herself","him","himself","his","how","how's",
    "i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it",
    "it's","its","itself","let's","me","more","most","mustn't","my",
    "myself","no","nor","not","of","off","on","once","only","or","other",
    "ought","our","ours","ourselves","out","over","own","same","shan't",
    "she","she'd","she'll","she's","should","shouldn't","so","some","such",
    "than","that","that's","the","their","theirs","them","themselves","then",
    "there","there's","these","they","they'd","they'll","they're","they've",
    "this","those","through","to","too","under","until","up","very","was",
    "wasn't","we","we'd","we'll","we're","we've","were","weren't","what",
    "what's","when","when's","where","where's","which","while","who",
    "who's","whom","why","why's","will","with","won't","would","wouldn't",
    "you","you'd","you'll","you're","you've","your","yours","yourself",
    "yourselves","said","also","s","re","ve","ll","t","d",
}


def preprocess_spark(spark, df):
    """PySpark path: RegexTokenizer → StopWordsRemover."""
    from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
    from pyspark.sql.functions import col

    # Ensure text column exists; fall back to 'title' if absent
    text_col = "text" if "text" in df.columns else df.columns[0]

    # Drop rows with null text
    df = df.filter(col(text_col).isNotNull())

    tokenizer = RegexTokenizer(
        inputCol=text_col,
        outputCol="tokens",
        pattern=r"[^a-zA-Z]+",   # split on anything that is not a letter
        toLowercase=True,
        minTokenLength=2,
    )
    df = tokenizer.transform(df)

    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered_tokens",
    )
    df = remover.transform(df)

    return df, text_col


def preprocess_pandas(df):
    """Pandas path: regex split + lowercase → stopword filter."""
    import re

    text_col = "text" if "text" in df.columns else df.columns[0]

    df = df.dropna(subset=[text_col]).copy()
    df["tokens"] = (
        df[text_col]
        .str.lower()
        .apply(lambda s: [t for t in re.split(r"[^a-zA-Z]+", s) if len(t) >= 2])
    )
    df["filtered_tokens"] = df["tokens"].apply(
        lambda toks: [t for t in toks if t not in _STOPWORDS]
    )
    return df, text_col


def preprocess(spark, df):
    if USE_SPARK and spark is not None:
        df, text_col = preprocess_spark(spark, df)
        print("[OK]  Preprocessing done (PySpark)")
    else:
        df, text_col = preprocess_pandas(df)
        print("[OK]  Preprocessing done (pandas)")
    return df, text_col


def show_preprocessed(df, text_col):
    """Print 5 rows showing original text, tokens, filtered_tokens."""
    cols = [text_col, "tokens", "filtered_tokens"]
    print("\n── Preprocessed sample (5 rows) ────────────────")
    if USE_SPARK:
        df.select(*cols).show(5, truncate=100)
    else:
        display_df = df[cols].head(5).copy()
        # Truncate long strings for readability
        display_df[text_col] = display_df[text_col].str[:80]
        display_df["tokens"]          = display_df["tokens"].apply(lambda x: x[:8])
        display_df["filtered_tokens"] = display_df["filtered_tokens"].apply(lambda x: x[:8])
        print(display_df.to_string(index=False))


# ─── 5. Feature Engineering ───────────────────────────────────────────────────

def feature_engineering_spark(df):
    """Returns dict: {'tfidf': (df, col, build_time), 'hash': (df, col, build_time)}"""
    from pyspark.ml.feature import HashingTF, IDF, CountVectorizer

    # ── TF-IDF via CountVectorizer + IDF ──────────────────────────────────────
    t0       = time.time()
    cv       = CountVectorizer(inputCol="filtered_tokens", outputCol="tf_raw",
                               vocabSize=50_000, minDF=2.0)
    cv_model = cv.fit(df)
    df_tf    = cv_model.transform(df)
    idf       = IDF(inputCol="tf_raw", outputCol="tfidf_features")
    idf_model = idf.fit(df_tf)
    df_tfidf  = idf_model.transform(df_tf)
    tfidf_time = time.time() - t0

    # ── HashingTF + IDF (FIXED: IDF now applied after HashingTF) ──────────────
    t0          = time.time()
    htf         = HashingTF(inputCol="filtered_tokens", outputCol="hash_raw",
                            numFeatures=50_000)
    df_htf      = htf.transform(df)
    idf_h       = IDF(inputCol="hash_raw", outputCol="hash_features")
    idf_h_model = idf_h.fit(df_htf)
    df_hash     = idf_h_model.transform(df_htf)
    hash_time   = time.time() - t0

    print(f"[OK]  TF-IDF build: {tfidf_time:.2f}s | "
          f"HashingTF+IDF build: {hash_time:.2f}s (PySpark)")
    return {"tfidf": (df_tfidf, "tfidf_features", tfidf_time),
            "hash":  (df_hash,  "hash_features",  hash_time)}


def feature_engineering_pandas(df):
    """Returns dict: {'tfidf': (X, y, build_time), 'hash': (X, y, build_time)}"""
    from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

    corpus = df["filtered_tokens"].apply(lambda toks: " ".join(toks))
    y      = df["label"].values

    t0        = time.time()
    tfidf_vec = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
    X_tfidf   = tfidf_vec.fit_transform(corpus)
    tfidf_time = time.time() - t0
    feat_dim  = X_tfidf.shape[1]

    t0       = time.time()
    hash_vec = HashingVectorizer(n_features=50_000, alternate_sign=False)
    X_hash   = hash_vec.fit_transform(corpus)
    hash_time = time.time() - t0

    print(f"[OK]  TF-IDF build: {tfidf_time:.2f}s (dim={feat_dim:,}) | "
          f"HashingTF build: {hash_time:.2f}s (pandas)")
    return ({"tfidf": (X_tfidf, y, tfidf_time),
             "hash":  (X_hash,  y, hash_time)},
            feat_dim)


def feature_engineering(spark, df):
    if USE_SPARK and spark is not None:
        return feature_engineering_spark(df), "spark", None
    else:
        feats_dict, feat_dim = feature_engineering_pandas(df)
        return feats_dict, "pandas", feat_dim


# ─── 6. Model Training & Evaluation ───────────────────────────────────────────

def _fmt(v):
    return f"{v:.4f}"


# ── Spark path ─────────────────────────────────────────────────────────────────
def train_and_evaluate_spark(features_dict):
    from pyspark.ml.classification import (
        LogisticRegression as SparkLR,
        DecisionTreeClassifier as SparkDT,
        RandomForestClassifier as SparkRF,
    )
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    results        = []
    spark_total_t0 = time.time()

    for feat_name, (df_feat, feat_col, build_time) in features_dict.items():
        train_df, test_df = df_feat.randomSplit([0.8, 0.2], seed=42)
        train_df.cache()
        test_df.cache()

        # FIXED: direct instantiation per feat_col — no extractParamMap()
        # FIXED: numTrees 50 → 100 to match draft spec
        model_defs = [
            ("LR", SparkLR(featuresCol=feat_col, labelCol="label", maxIter=20)),
            ("DT", SparkDT(featuresCol=feat_col, labelCol="label", maxDepth=10)),
            ("RF", SparkRF(featuresCol=feat_col, labelCol="label",
                           numTrees=100, maxDepth=10)),
        ]

        for model_name, model in model_defs:
            t0         = time.time()
            fitted     = model.fit(train_df)
            train_time = time.time() - t0
            preds      = fitted.transform(test_df)

            def _eval(metric, _preds=preds):
                return MulticlassClassificationEvaluator(
                    labelCol="label", predictionCol="prediction",
                    metricName=metric
                ).evaluate(_preds)

            results.append({
                "Model":     model_name,
                "Feature":   feat_name.upper(),
                "BuildTime": build_time,
                "Accuracy":  _eval("accuracy"),
                "Precision": _eval("weightedPrecision"),
                "Recall":    _eval("weightedRecall"),
                "F1":        _eval("f1"),
                "TrainTime": train_time,
            })
            print(f"  [done] {model_name} + {feat_name.upper()} "
                  f"(train={train_time:.1f}s)")

        train_df.unpersist()
        test_df.unpersist()

    spark_total = time.time() - spark_total_t0
    print(f"\n[Comparison 1] Spark total training time: {spark_total:.2f}s")
    return results


# ── Pandas / sklearn path ──────────────────────────────────────────────────────
def train_and_evaluate_pandas(features_dict):
    from sklearn.linear_model    import LogisticRegression as SkLR
    from sklearn.tree            import DecisionTreeClassifier as SkDT
    from sklearn.ensemble        import RandomForestClassifier as SkRF
    from sklearn.model_selection import train_test_split
    from sklearn.metrics         import (accuracy_score, precision_score,
                                         recall_score, f1_score)

    # FIXED: n_estimators 50 → 100 to match draft spec
    model_defs = [
        ("LR", SkLR(max_iter=1000, solver="saga", n_jobs=-1)),
        ("DT", SkDT(max_depth=10, random_state=42)),
        ("RF", SkRF(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
    ]

    results          = []
    sklearn_total_t0 = time.time()

    for feat_name, (X, y, build_time) in features_dict.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        for model_name, model in model_defs:
            t0         = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            y_pred     = model.predict(X_test)

            results.append({
                "Model":     model_name,
                "Feature":   feat_name.upper(),
                "BuildTime": build_time,
                "Accuracy":  accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred,
                                             average="weighted", zero_division=0),
                "Recall":    recall_score(y_test, y_pred,
                                          average="weighted", zero_division=0),
                "F1":        f1_score(y_test, y_pred,
                                      average="weighted", zero_division=0),
                "TrainTime": train_time,
            })
            print(f"  [done] {model_name} + {feat_name.upper()} "
                  f"(train={train_time:.1f}s)")

    sklearn_total = time.time() - sklearn_total_t0
    print(f"\n[Comparison 1] Sklearn total training time: {sklearn_total:.2f}s")
    return results






def train_and_evaluate(spark, features_dict, mode):
    print("\n── Training models ─────────────────────────────────────────────────")
    if mode == "spark":
        results = train_and_evaluate_spark(features_dict)
    else:
        results = train_and_evaluate_pandas(features_dict)
    return results


# ─── 7. Results table ─────────────────────────────────────────────────────────

def print_results_table(results):
    header = (f"{'Model':<6} {'Feature':<8} {'Accuracy':>10} {'Precision':>10}"
              f" {'Recall':>10} {'F1':>10} {'TrainTime(s)':>14}")
    sep    = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['Model']:<6} {r['Feature']:<8}"
            f" {_fmt(r['Accuracy']):>10}"
            f" {_fmt(r['Precision']):>10}"
            f" {_fmt(r['Recall']):>10}"
            f" {_fmt(r['F1']):>10}"
            f" {r['TrainTime']:>13.2f}s"
        )
    print(sep)

    # Best model
    best = max(results, key=lambda x: x["F1"])
    print(f"\n★  Best model by F1 : {best['Model']} + {best['Feature']}"
          f"  →  F1={_fmt(best['F1'])}  Acc={_fmt(best['Accuracy'])}"
          f"  TrainTime={best['TrainTime']:.2f}s")

    # Feature build time summary
    print("\n── Feature Build Times ───────────────────────────────")
    seen = set()
    for r in results:
        key = r["Feature"]
        if key not in seen:
            print(f"  {key:<8}: {r['BuildTime']:.2f}s")
            seen.add(key)


