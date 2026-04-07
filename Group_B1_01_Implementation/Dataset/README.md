# Dataset Link

We are using a publicly available Fake News Dataset since it is too large to commit to GitHub.

**Dataset Name:** Clement Bisaillon's Fake and Real News Dataset
**Download Link:** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## Instructions
1. Download the `.csv` file from the link above.
2. Place the downloaded `.csv` inside this `Dataset/` folder.
3. The `main.ipynb` will read it directly from here using `spark.read.csv('Dataset/your-dataset.csv', header=True)`.
