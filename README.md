
# 🧠 Breast Cancer Diagnosis (Logistic Regression)

Machine Learning project to diagnose breast cancer (malignant/benign) using logistic regression, trained on the Breast Cancer Wisconsin dataset.

<p align="center">
  <a href="https://github.com/TechFreak2003/breast-cancer-logreg/issues"><img src="https://img.shields.io/github/issues/TechFreak2003/breast-cancer-logreg"></a> 
  <a href="https://github.com/TechFreak2003/breast-cancer-logreg/stargazers"><img src="https://img.shields.io/github/stars/TechFreak2003/breast-cancer-logreg"></a>
  <a href="https://github.com/TechFreak2003/breast-cancer-logreg/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg">
  </a>
</p>

<p align="center">
  <a href="#-features">Features</a> |
  <a href="#%EF%B8%8F-tech-stack">Tech Stack</a> |
  <a href="#-installation">Installation</a> |
  <a href="#-project-structure">Project Structure</a> |
  <a href="#-contributing">Contributing</a> |
  <a href="#%EF%B8%8F-author">Author</a>
</p>

## 🌟 Features

- Logistic Regression classifier for tumor diagnosis
- Exploratory data analysis (EDA) and preprocessing
- Confusion matrix, ROC curve, and model evaluation metrics
- Modular codebase (data, preprocessing, model, utils)
- Easy reproducibility and extendability for other ML models

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **ML Library**: scikit-learn
- **Data Handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Notebook Support**: Jupyter

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Recommended: virtual environment (venv or conda)

### Steps

```bash
# Clone the repository
git clone https://github.com/TechFreak2003/breast-cancer-logreg.git
cd breast-cancer-logreg

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py
```

## 📁 Project Structure

```
breast-cancer-diagnosis/
├── data/
│   └── breast_cancer_data.csv     # Dataset (Kaggle)
│
├── notebooks/
│   └── exploratory_analysis.ipynb # EDA and initial experiments
│
├── src/
│   ├── data_loader.py             # Load and read CSV
│   ├── preprocess.py              # Data cleaning & encoding
│   ├── model.py                   # Train & evaluate logistic regression
│   └── utils.py                   # Visualization & helpers
├── .gitignore
├── README.md
├── requirements.txt
└── main.py                        # Orchestrates the full pipeline
```

## 📊 Dataset

- **Source**: [Kaggle - Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Target Column**: `diagnosis` (`M` = malignant, `B` = benign)
- **Features**: 30 numerical features (radius, texture, area, etc.)

## 📈 Results

Example outputs include:

- **Accuracy**: 95%+
- **Confusion Matrix**: True Positives / False Negatives breakdown
- **ROC-AUC**: Area under the curve visualized

> 📌 Include your actual metrics or screenshots after model evaluation

## 👥 Contributing

Contributions are welcome! Please check the [issues](https://github.com/TechFreak2003/breast-cancer-logreg/issues) and submit PRs to improve the code, add enhancements, or fix bugs.

To contribute:

1. Fork the repo
2. Create a new branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push and open a Pull Request

## 👨‍💻 Contributors

| Avatar | Name | GitHub | Role | Contributions |
|--------|------|--------|------|---------------|
| <img src="https://github.com/TechFreak2003.png" width="50px" height="50px" /> | Suvrodeep Das | [TechFreak2003](https://github.com/TechFreak2003) | Creator | Full implementation, docs |

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙋‍♂️ Author

Created with ❤️ by [Suvrodeep Das](https://suvrodeepdas.dev)
