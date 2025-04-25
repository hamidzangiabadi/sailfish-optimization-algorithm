
---

# 🚀 sailfishOptimizer

A Python-based feature selection tool using the **Sailfish Optimizer (SFO)** and other evolutionary algorithms, with **SVM** as the classifier. This project compares the performance of multiple metaheuristic algorithms on different datasets for feature selection.

---

## 📌 Features

- Feature selection using:
  - Sailfish Optimizer (SFO)
  - Genetic Algorithm (GA)
  - Whale Optimization Algorithm (WOA)
  - Elephant Herding Optimization (EHO)
  - Grey Wolf Optimizer (GWO)
  - Moth-Flame Optimizer (MFO)
- Objective function combining feature reduction and accuracy
- Evaluation via **SVM** with linear kernel
- Dataset handling and cross-validation
- Results saving for comparison

---

## 🛠️ Requirements

- Python 3.7+
- `numpy`
- `scikit-learn`
- [`mealpy`](https://pypi.org/project/mealpy/)

Install dependencies using:

```bash
pip install numpy scikit-learn mealpy
```

---

## 🧠 How It Works

1. **Data Preparation:**  
   Datasets are split into train and test sets. Data is normalized using standard preprocessing.

2. **Binary Feature Selection:**  
   Real-valued outputs from the optimizers are converted into binary feature masks.

3. **Objective Function:**  
   ```python
   cost = (alpha * (1 - accuracy)) + (beta * (num_features_selected / total_features))
   ```
   Where `alpha` and `beta` balance accuracy vs. feature reduction.

4. **Optimization Algorithms:**  
   Each algorithm runs multiple times to solve the same optimization problem for comparison.

5. **Classification:**  
   A Support Vector Machine (SVM) is trained on the selected features and accuracy is measured.

---

## 🔁 Example Workflow

```python
SFOModel = SFO.OriginalSFO(epoch=30, pop_size=50, pp=0.1, AP=4, epsilon=0.01)
g_best = SFOModel.solve(problem_dict)
save_results(g_best, "SFO", num_features, dataset["name"])
```

Repeat similar steps for GA, EHO, WOA, GWO, and MFO.

---

## 📊 Results

The accuracy and number of features selected are recorded for each algorithm and dataset. You can modify the `save_results` function to log results as CSV or visualize them.

---
