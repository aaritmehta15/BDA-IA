# Big Data Analytics (BDA): Distributed Fake News Detection

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Spark](https://img.shields.io/badge/Apache_Spark-Distributed-orange) ![Turnitin](https://img.shields.io/badge/Turnitin_Similarity-7%25-success)

This repository contains the complete execution framework, codebase, and research collateral for our **Big Data Analytics (IA2 & Lab CA)** academic project. Our core objective was to empirically demonstrate how migrating traditional monolithic Machine Learning into a **Distributed Big Data Framework (Apache Spark)** drastically resolves computational bottlenecks and memory limitations when handling massive volumes of unstructured text.

<br>

---

## 📚 1. Extensive Literature Review & Research Motive

Our project's architecture is the direct result of a vast, systematic literature survey. We reviewed **26 pivotal research papers** in the domain of fake news detection to identify systemic flaws in current implementations.

**Key Findings from our Literature Survey:**
- **The Memory Bottleneck:** Traditional frameworks (like `scikit-learn`) rely heavily on single-threaded `CountVectorizers` and `TF-IDF` matrices. As the corpus grows, scanning the entire vocabulary to build feature maps causes immediate Out-Of-Memory (OOM) errors. (**See Paper_Summaries.md for full paper breakdowns**).
- **The Scalability Gap:** While deep learning approaches achieve high accuracy, they scale poorly outside of expensive GPU clusters.
- **The Solution:** We concluded that applying the **Hadoop MapReduce Paradigm** via **Apache Spark** is the most optimal way to handle text stream scalability. PySpark naturally partitions data (similar to HDFS blocks) and computes features locally on distributed nodes using `HashingTF`—entirely eliminating the vocabulary memory limit.

All 26 analyzed research papers, including their methodologies, working systems, and identified gaps, are meticulously mapped in the provided `Literature_Review_Sheet.xlsx` and linked in `resources.doc`.

<br>

---

## ⚙️ 2. Proposed Architecture & Code Implementation

Based on our research, we built a fully distributed Machine Learning pipeline. 

### Core Codebase: `Group_B1_01_Implementation/main.ipynb`

Our implementation is a Jupyter Notebook built to be executed on **Google Colab** (which provides a simulated `local[*]` distributed engine).
Instead of limiting data, our script simulates Hadoop parallel processing across the following layers:

1. **Distributed Data Ingestion (Layer 1):** Loading full Kaggle datasets into partitioned PySpark DataFrames instead of static Pandas.
2. **Parallel Preprocessing (Layer 2):** Utilizing Spark's `RegexTokenizer` and `StopWordsRemover` simultaneously across all CPU worker nodes.
3. **Memory-Efficient Feature Engineering (Layer 3):** Discarding RAM-heavy vectorizers in favor of PySpark's `HashingTF` algorithm (which hashes words mathematically on the fly, rendering the memory requirement to `O(1)`).
4. **Ensemble Machine Learning (Layers 4 & 5):** Because the Spark cluster is highly scalable, we are afforded the computational budget to train aggressive ensemble models (like a 50-tree **Random Forest Classifier**) that would normally crash a single-threaded server.

<br>

---

## 📊 3. Empirical Results & Output Visualizations

By comparing a restricted traditional `sklearn` baseline against our distributed `PySpark` pipeline on the exact same test-splits, we achieved highly defensible empirical results. The pipeline generates several automated output graphics saved into the **`Group_B1_01_Implementation/Output/`** folder.

### What We Achieved (Code Outputs):

1. **Massive Speed & Scalability Gains (`figure3_training_time.png` & `figure4_feature_build_time.png`):**
   - **Result:** The code explicitly outputted metrics proving that while traditional models scale linearly (taking progressively longer per record), PySpark offloads calculations to partitioned tasks. Our Big Data pipeline drastically reduced feature building time simply by utilizing horizontal core scaling.

2. **Uncompromising Accuracy (`figure1_confusion_matrix.png`):**
   - **Result:** We achieved a highly stable **99%+ F1-Score**. The generated Confusion Matrix demonstrates that the PySpark Random Forest model perfectly balanced True Positives and True Negatives without suffering from the false-positive bias typical of simplistic TF-IDF Logistic Regressions.

3. **Classification Robustness (`figure2_roc_auc.png`):**
   - **Result:** The ROC Area Under Curve (AUC) output graphs explicitly display near-perfect classification performance across various threshold boundaries.

*All graphical outputs are automatically saved to the local `Output` folder when you run the pipeline.*

All methodology, mathematical frameworks (Hadoop, Sqoop, Spark schemas), architecture diagrams, performance metrics, and social impact statements are comprehensively laid out in our primary submission report: **`Technical_Chapter/Group_BDA_IA.pdf`**.

<br>

---

## 🚀 How to Run the Infrastructure

### Prerequisites
Install the required packages strictly from our root configuration:
```bash
pip install -r requirements.txt
```
*(Note for local execution: Running PySpark locally on Windows architectures requires Java 11/17 to be installed securely and the `JAVA_HOME` path active).*

### Running via Google Colab (Recommended for Academic Review)
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `Group_B1_01_Implementation/main.ipynb`.
3. Select **Run All**. The notebook will dynamically provision standard resources, connect to KaggleHub to download the dataset silently, and execute the comparative Big Data metrics layer by layer.

<br>

---
*Group B1_01 — Aarit Mehta, Akshat Panchal, Affan Shaikh*
