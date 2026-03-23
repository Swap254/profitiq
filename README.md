# ProfitIQ — Startup Profit Prediction

A machine learning project that predicts startup profit based on R&D Spend, Administration Cost, and Marketing Spend using multiple regression algorithms.

## Features

- Compares 5 regression models side by side
- Correlation analysis to identify strongest predictors
- Feature importance visualization (Random Forest)
- Clean performance metrics (R², MAE, RMSE)
- Saves charts as PNG outputs

## Tech Stack

`Python` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn`

## Project Structure

```
profitiq/
├── profit_prediction.py    # Main script — train, evaluate, visualize
├── 50_Startups.csv         # Dataset (50 startup companies)
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/swapnilchitalkar/profitiq.git
cd profitiq
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the script
```bash
python profit_prediction.py
```

## Output

```
── Model Performance ──
Model                      R² Score          MAE         RMSE
──────────────────────────────────────────────────────────────
Random Forest              0.9521      4821.33      7102.45
Linear Regression          0.9347      5234.21      8021.34
Ridge Regression           0.9341      5289.12      8055.67
Decision Tree              0.9102      6012.44      9234.56
Lasso Regression           0.8923      6512.33      9823.44

✅ Best Model: Random Forest with R² = 0.9521
```

### Generated Charts
| File | Description |
|---|---|
| `correlation_heatmap.png` | Feature correlation matrix |
| `model_comparison.png` | R² score comparison across models |
| `feature_importance.png` | Random Forest feature importance |

## Key Findings

- **R&D Spend** is the strongest predictor of profit (highest correlation)
- **Random Forest** achieves the best R² score (~95%)
- **Linear Regression** performs surprisingly well due to strong linear relationship between R&D Spend and Profit

## Dataset

50 startup companies with the following features:

| Column | Description |
|---|---|
| R&D Spend | Amount spent on research & development |
| Administration | Administrative costs |
| Marketing Spend | Amount spent on marketing |
| Profit | Target variable |

## Author

**Swapnil Chitalkar**
[LinkedIn](https://www.linkedin.com/in/swapnil-chitalkar)
