# 🌳 Decision Tree Classifier from Scratch (Python)

A **from-scratch implementation of a binary decision tree classifier** using the **CART algorithm** and **Gini impurity**, built for educational purposes during my summer break.

This project avoids high-level ML libraries (e.g. `sklearn`) and focuses on **understanding how decision trees actually work internally**: splitting logic, impurity computation, recursion, and prediction traversal.

---

## 🧠 Theory & Development Process

This implementation is the result of a deep dive into Information Theory and CART methodology.
My process involved manually deriving the mathematics of entropy and impurity to ensure algorithmic correctness.

Key decisions reflected in the code:

- **Binary splits only**  
  Each node splits the data into exactly two subsets, following the CART standard.

- **Greedy local optimization**  
  At each node, the split minimizing the *local* weighted Gini impurity is selected, without backtracking.

- **Threshold generation strategy**  
  - Numerical features are split using midpoints between sorted unique values  
  - Categorical features are split using equality checks

- **Leaf construction via majority class**  
  When no further split is beneficial, the node predicts the class with the highest empirical probability.

- **Stopping conditions derived from theory**  
  Tree growth halts when impurity reaches zero or when further splits provide no informational gain.


## 📐 The Mathematical Core

The model partitions data by minimizing the "randomness" or impurity within a given set

### Gini Index
This is the primary metric used in the code to create binary splits.
An attribute with a low Gini index is always preferred over a high one.

$$Gini = 1 - \sum_{i=1}^{C} (p_i)^2$$

Where $$p_i$$ is the probability of an element belonging to a specific class.

### Entropy Foundation
My research also covered Shannon Entropy, which measures information content. 
While this implementation uses Gini, the logic is rooted in the same principles of minimizing disorder.

$$H(X) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

## 🔬 From Theory to Code

The theoretical concepts map directly to implementation details:

- **Entropy / Gini** → `_gini_impurity`, `_weighted_gini`
- **Information minimization** → `_find_best_split`
- **Recursive partitioning** → `_build_tree`
- **Decision process** → `Node.get_next_node`
- **Prediction as tree traversal** → `predict`

--- 

## ✨ Features

- Binary **classification tree**
- **CART-style splitting** using **Gini impurity**
- Supports:
  - Numerical features (`<= threshold`)
  - Categorical features (`== category`)
- Recursive tree construction
- Configurable maximum depth
- Tree inspection via `print_tree()`
- Clean, object-oriented design (`Tree` + `Node`)

---

## 🧠 Algorithm Overview

The model follows the **CART (Classification And Regression Tree)** methodology:

1. Iterate through all features
2. Try all possible split thresholds
3. Compute **weighted Gini impurity**
4. Choose the split that minimizes impurity
5. Recursively build left and right subtrees
6. Stop when the node is pure or max depth is reached

---

## 🚀 Usage

```python
from classification_binaryTree import classification_binaryTree

clf = classification_binaryTree(maxDepth=5)

```

### Fit the Model

```python
model.fit(X_train, y_train)
```

- `X_train`: pandas DataFrame  
- `y_train`: pandas Series  

### Make Predictions

```python
y_pred = model.predict(X_test)
```

### Print the Tree

```python
model.print_tree()
```

---

## 🧱 Implementation Highlights

- Recursive tree construction (`_build_tree`)
- Greedy split selection
- Explicit node representation
- Majority-class leaf prediction

---

## ⚠️ Limitations

This is an **educational implementation**:

- No pruning
- No missing-value handling
- No vectorized prediction
- No ensemble methods

---

## 🎓 Motivation

This project was built during my summer break to deeply understand **decision trees from first principles**, without relying on ML libraries.

It now serves as a conceptual foundation for further studies in Machine Learning and Artificial Intelligence.

---

## 📚 References & Inspiration

- Bamford, M., “Decision Trees Explained”  
  https://medium.com/@MrBam44/decision-trees-91f61a42c724

---

## 👤 Author

**Bogdan-Vlad Gurău**  
Computer Engineering student (UTCN/CTI)
