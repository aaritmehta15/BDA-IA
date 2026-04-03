# Fake News Detection — Research Paper Summaries

---

## Paper 1: Fake News Detection on Social Media: A Data Mining Perspective
**Authors:** Kai Shu, Amy Sliva, Suhang Wang, Jiliang Tang, Huan Liu

| Field | Details |
|---|---|
| Type | Survey / Framework |
| Methodology | Content analysis, social context, user engagement, propagation graphs |
| Architecture | News text → user interaction → propagation network → ML classification |
| What Worked | Content + social signals combined; propagation graph analysis |
| What Failed | Pure NLP, keyword-based, rule-based methods |
| Gap | No real-time scalable systems; multimodal data ignored; noisy incomplete streams |
| Big Data Scope | Real-time Twitter streaming, Spark GraphX / Hadoop graph mining |

---

## Paper 2: Big Data ML-Based Fake News Detection using Distributed Learning
**Authors:** Alaa Altheneyan, Aseel Alhadlaq

| Field | Details |
|---|---|
| Type | Experimental / Big Data Architecture |
| Methodology | Apache Spark, stacked ensemble, N-gram, TF-IDF, Count Vectorizer, HashingTF |
| Architecture | Distributed Spark nodes → feature engineering → ensemble classification |
| What Worked | F1 = 92.45% vs baseline 83.10% (+9.35% gain) |
| What Failed | Single classifiers (LR, Decision Tree) had lower performance |
| Gap | Text-only; no image/video; no multimodal or real-time streaming |
| Big Data Scope | Scalability, training speed, distributed feature extraction |

---

## Paper 3: A Review of Hadoop-Based Frameworks for Fake Review Detection
**Authors:** Zainab Barwaniwala, Parth Chandna, Abhinav Goud, Dr. Ramesh S.

| Field | Details |
|---|---|
| Type | Review |
| Methodology | Text mining, sentiment analysis, ML classification on Hadoop |
| Architecture | HDFS storage → MapReduce → distributed ML |
| What Worked | Scalability over millions of reviews; parallel processing |
| What Failed | Keyword systems failed for sarcasm, short deceptive reviews |
| Gap | No transformers, no BERT, no GNN |
| Big Data Scope | Speed, large-scale review analysis, batch processing |

---

## Paper 4: Dataset of Fake News Detection and Fact Verification: A Survey
**Author:** Taichi Murayama

| Field | Details |
|---|---|
| Type | Dataset Survey |
| Methodology | Reviewed 118 datasets across fake news, fact verification, rumor detection |
| What Worked | Dataset taxonomy and benchmark comparison |
| Gap | Dataset bias, domain imbalance, multilingual scarcity |
| Big Data Scope | Distributed dataset indexing, data lake architecture, multimodal warehousing |

---

## Paper 5: Fake News Detection: It's All in the Data!
**Authors:** Soveatin Kuntur, Anna Wróblewska, Marcin Paprzycki, Maria Ganzha

| Field | Details |
|---|---|
| Type | Dataset Quality Analysis |
| Methodology | Dataset comparison, label analysis, bias analysis, multimodal review |
| Key Finding | Data quality impacts model more than algorithm choice |
| Gap | Lacks model experimentation depth |
| Big Data Scope | Scalable data labeling, distributed annotation, bias-aware balancing |

---

## Paper 6: Scalable Fake News Detection: NLP and Embedding Models for Large-Scale Data
**Authors:** Dr. Surjeet et al.

| Field | Details |
|---|---|
| Type | Experimental |
| Methodology | BoW, TF-IDF, Word2Vec, BERT, LR, Random Forest, Neural Networks on Hadoop/Spark |
| What Worked | BERT outperformed traditional embeddings |
| What Failed | BoW and TF-IDF weak for context |
| Gap | High computational cost |
| Big Data Scope | Scalability, batch inference, large-scale training |

---

## Paper 7: FAKEDETECTOR: Effective Fake News Detection with Deep Diffusive Neural Network
**Authors:** Jiawei Zhang, Bowen Dong, Philip S. Yu

| Field | Details |
|---|---|
| Type | Experimental / Graph-based |
| Methodology | Deep Diffusive Neural Network, graph diffusion, credibility propagation |
| Architecture | Heterogeneous graph: news articles + creators + subjects |
| What Worked | Relational learning; detects creator credibility |
| Gap | Needs large graph datasets and high compute |
| Big Data Scope | Spark GraphX, graph databases, distributed GNN training |

---

## Paper 8: Fake News Detection Using Deep Learning: A Systematic Literature Review
**Authors:** Mohammad Q. Alnabhan, Paula Branco

| Field | Details |
|---|---|
| Type | SLR |
| Methodology | Reviewed CNN, LSTM, RNN, transformers, transfer learning |
| What Worked | DL models outperform traditional ML |
| Gap | Class imbalance, poor transfer learning, lack of generalization |
| Big Data Scope | Distributed DL pipelines on Spark / distributed GPUs |

---

## Paper 9: Automatic Detection of Fake News
**Authors:** Veronica Pérez-Rosas, Bennett Kleinberg, Alexandra Lefevre, Rada Mihalcea

| Field | Details |
|---|---|
| Type | Experimental / NLP |
| Methodology | Linguistic features, grammar, writing style, stylometric features |
| What Worked | Accuracy up to 78% |
| What Failed | Limited generalization across domains |
| Gap | Needs social and propagation features |
| Big Data Scope | Large corpus learning, cross-domain scalable training |

---

## Paper 10: A Scoping Review of Big Data Analytics with Context-Based Fake News Detection
**Authors:** Khurram Shahzad, Shakeel Ahmad Khan, Shakil Ahmad, Abid Iqbal

| Field | Details |
|---|---|
| Type | Scoping Review (42 papers) |
| Methodology | Literature review, content analysis, trend identification, challenge mapping |
| Key Finding | Quality big data has strong positive correlation with detection performance |
| What Failed | Traditional systems failed on unstructured, high-velocity, biased data |
| Gap | Poor authentic dataset generation; multilingual misinformation; context-aware detection |
| Big Data Scope | Real-time analytics, social network graph mining, context extraction |

---

## Paper 11: Big Data and Quality Data for Fake News and Misinformation Detection
**Authors:** Fatemeh Torabi Asr, Maite Taboada

| Field | Details |
|---|---|
| Type | NLP + Dataset |
| Methodology | Text classification, topic modeling, dataset quality analysis; introduced MisInfoText |
| Key Finding | "Fake news is a big data problem being solved with small data" |
| What Failed | Insufficient training data, topic imbalance, dataset bias |
| Gap | Need balanced, cross-topic, scalable labeled repositories |
| Big Data Scope | Distributed data collection, web-scale labeling, large-scale topic balancing |

---

## Paper 12: Fake News Data Exploration and Analytics
**Authors:** Mazhar Javed Awan et al.

| Field | Details |
|---|---|
| Type | Experimental / ML |
| Methodology | TF-IDF, LR, Random Forest, Decision Tree |
| Results | TF-IDF 99.52%, LR 98.63%, RF 99.63%, DT 99.68% |
| What Failed | Possible overfitting; limited real-world generalization |
| Gap | No DL, no streaming, no propagation modeling |
| Big Data Scope | Spark MLlib, distributed TF-IDF, large social stream datasets |

---

## Paper 13: The Power of Big Data Analytics over Fake News
**Authors:** Enrique Cano-Marin et al.

| Field | Details |
|---|---|
| Type | Scientometric + Systematic Review |
| Methodology | Bibliometric analysis, Twitter analytics, healthcare misinformation |
| What Worked | Healthcare/COVID misinformation detection, public sentiment |
| Gap | Needs predictive rather than retrospective detection |
| Big Data Scope | Twitter big data useful for predictive fake news analytics |

---

## Paper 14: Comparative Evaluation of BERT-like Models and LLMs
**Authors:** Shaina Raza, Drai Paulen-Patterson, Chen Ding

| Field | Details |
|---|---|
| Type | Experimental / Advanced NLP |
| Methodology | BERT, LLMs, GPT-4 labeling, human validation; 10,000 articles |
| What Worked | BERT outperforms LLMs in classification; LLMs stronger in robustness |
| Gap | High computational cost; LLM inference expensive |
| Big Data Scope | AI-generated labeling at scale, large-scale fine-tuning pipelines |

---

## Paper 15: Fake News Early Detection
**Authors:** Xinyi Zhou et al.

| Field | Details |
|---|---|
| Type | Experimental / NLP |
| Methodology | Lexicon, syntax, semantic, discourse-level feature engineering |
| What Worked | Early detection before propagation |
| Gap | No multimodal integration |
| Big Data Scope | Streaming text systems, early real-time social feeds |

---

## Paper 16: LIAR Benchmark Dataset
**Author:** William Yang Wang

| Field | Details |
|---|---|
| Type | Dataset + Hybrid CNN |
| Dataset | 12.8K manually labeled statements |
| What Worked | Foundational benchmark dataset |
| Gap | Short text statements only |
| Big Data Scope | Large fact-check pipelines, streaming political claims |

---

## Paper 17: Optimization and Improvement using Deep Learning
**Authors:** Tavishee Chauhan, Hemant Palivela

| Field | Details |
|---|---|
| Type | Experimental / DL |
| Methodology | LSTM, GloVe embeddings, tokenization, N-grams |
| What Worked | Accuracy: 99.88% |
| Gap | May not generalize across domains |
| Big Data Scope | Distributed LSTM training, Spark NLP |

---

## Paper 18: Analytics and Visualization of Detecting Fake News Accuracy
**Authors:** Stephanie Molina, Seongyong Hong

| Field | Details |
|---|---|
| Type | Experimental / Visualization |
| Methodology | Python + NLP + factuality score + source comparison + graphical visualization |
| What Worked | High interpretability |
| Gap | Less scalable; more academic prototype |
| Big Data Scope | Dashboard + real-time news stream analytics |

---

## Paper 19: Scalable Framework for Multilevel Streaming Data Analytics
**Authors:** Shihao Ge et al.

| Field | Details |
|---|---|
| Type | Big Data Architecture |
| Methodology | Spark Streaming, LSTM, SQL analytics |
| Architecture | Streaming ingestion → real-time processing → LSTM sentiment → SQL query |
| What Worked | Real-time analytics, large news stream handling |
| Gap | Not directly a fake news classifier |
| Big Data Scope | Best paper for streaming big data architecture |

---

## Paper 20: Decaffe: DHT Tree-Based Online Federated Fake News Detection
**Authors:** Cheng-Wei Ching, Liting Hu

| Field | Details |
|---|---|
| Type | Advanced / Federated Learning |
| Methodology | Federated learning, DHT tree, decentralized aggregation |
| Architecture | Root → branches → leaves (tree-based distributed) |
| What Worked | Mobile social networks, real-time detection, privacy-preserving |
| Gap | Complex deployment; synchronization overhead |
| Big Data Scope | One of the best scalable architecture papers; ideal for future work |

---

## Paper 21: Analysis of Classifiers for Fake News Detection
**Authors:** Vasu Agarwala, H. Parveen Sultana, Srijan Malhotra, Amitrajit Sarkar

| Field | Details |
|---|---|
| Type | Experimental / Classical ML |
| Methodology | BoW, N-grams, Count Vectorizer, TF-IDF + 5 classifiers |
| What Worked | TF-IDF representation; multi-classifier benchmarking |
| Gap | No DL, no social propagation, no streaming, poor scalability |
| Big Data Scope | Distributed TF-IDF, Spark ML, Twitter data ingestion |

---

## Paper 22: Fake News Detection: A Survey of Evaluation Datasets
**Authors:** Arianna D'Ulizia, Maria Chiara Caschera, Fernando Ferri, Patrizia Grifoni

| Field | Details |
|---|---|
| Type | Dataset Survey (27 datasets reviewed) |
| Methodology | Systematic literature review; 11 dataset characteristics identified |
| What Worked | Structured dataset taxonomy; benchmark comparison framework |
| Gap | No standardized datasets; poor multilingual/multimodal support |
| Big Data Scope | Large-scale benchmark repos, distributed data lake architecture |

---

## Paper 23: The Power of Fake News: Big Data Analysis of COVID-19
**Authors:** Sou Hyun Jang, Kyoung Eun Jung, Yong Jeong Yi

| Field | Details |
|---|---|
| Type | Big Data / Sentiment Analysis |
| Dataset | 38,057 comments; 98 fake-news-related articles |
| Methodology | Word2Vec, comment mining, big data sentiment analysis |
| What Worked | Sentiment mining, public reaction analysis, COVID misinformation trends |
| What Failed | Weak predictive fake-news classifier |
| Gap | Needs classification layer, misinformation forecasting, real-time dashboards |
| Big Data Scope | Large-scale comment mining significantly improved pattern detection |

---

## Paper 24: Leveraging Hadoop and Hybrid Deep Learning on Home Datasets
**Authors:** Asaad R. Kareem, Hasanen S. Abdullah

| Field | Details |
|---|---|
| Type | Big Data + DL Architecture |
| Methodology | Hadoop, CNN, LSTM, BI system; 4-phase architecture |
| What Worked | CNN + LSTM hybrid; Hadoop distributed processing; 512MB block size optimal |
| Gap | Not specifically fake-news focused |
| Big Data Scope | Ideal for big data implementation architecture discussion |

---

## Paper 25: Identification of Fake News Using ML in Distributed System
**Authors:** Mehruz Saif et al.

| Field | Details |
|---|---|
| Type | Experimental / Distributed ML |
| Methodology | PySpark, RDD, Spark Context; RF, FM, Linear SVC, LR |
| Architecture | Distributed ingestion → TF-IDF → Spark ML → distributed classifier |
| What Worked | Distributed learning, real-time scalable processing, multi-model comparison |
| Gap | No DL; no streaming Spark |
| Big Data Scope | Processing speed, storage scalability, parallel model training |

---

## Paper 26: A Review of Methodologies for Fake News Analysis
**Authors:** Mehedi Tajrian et al.

| Field | Details |
|---|---|
| Type | Review |
| Methodology | 4 perspectives: knowledge, style, propagation, source |
| What Worked | Strong theoretical framework; methodology mapping |
| Gap | Suggests Bayesian modeling as future work |
| Big Data Scope | Bayesian streaming learning, distributed probabilistic inference |

---
