# XGBoost

For an XGBoost model to work we need to train it by using a datatype known as DMatrix.
Internally, the DMatrix includes:

-The feature matrix (in compressed format)(basically, what are the values of the various feature columns in the dataset)

-Labels (targets)(outputs)

Optional things like:

-Weights (importance per sample)

-Base margins

-Feature names

-Missing value masks


## 🚀 **Core XGBoost Hyperparameters – Deep Dive**

---

### 🔷 1. `max_depth` — *How deep each tree can go*

#### 🔍 Intuition:

* Think of a tree asking questions like: “Is sepal length > 5.1?”
* `max_depth` controls how many such decisions it can stack.
* More depth = more complex decision rules

#### 🛠 Practical Use:

* Small datasets → `max_depth = 3–6`
* Large datasets → `max_depth = 6–10` (but watch for overfitting)

#### 🎯 Visual:

```python
max_depth=2
# Only 2 levels of splits: very simple trees, may underfit

max_depth=10
# Trees go very deep, can model complex patterns, but might memorize noise (overfit)
```

---

### 🔷 2. `eta` or `learning_rate` — *How fast to learn*

#### 🔍 Intuition:

* Each new tree tries to **fix the errors** of the previous trees.
* `eta` controls **how much of that fix is applied**.

#### 🛠 Practical Use:

* Lower `eta` = slower learning, but more stable.
* Common practice: `eta = 0.01 – 0.3`
* If using small `eta`, increase `num_boost_round`

#### 🎯 Visual:

```python
eta=0.3
# Big corrections → fast learning → risk of overshooting → overfitting

eta=0.01
# Small corrections → slow learning → needs more trees but generalizes better
```

---

### 🔷 3. `num_boost_round` — *Number of trees to grow*

#### 🔍 Intuition:

* Each tree tries to correct the mistakes of the previous ones.
* More trees = better accuracy (up to a point)

#### 🛠 Practical Use:

* With low `eta`, you’ll need more boosting rounds.

```python
# Tradeoff:
eta=0.3 → 50 rounds might be enough
eta=0.05 → might need 200+ rounds
```

---

### 🔷 4. `subsample` — *What % of rows to use per tree*

#### 🔍 Intuition:

* Like giving the model partial data each time (like Random Forests)
* Helps prevent overfitting

#### 🛠 Practical Use:

* Typical range: `0.5 – 1.0`

```python
subsample=1.0
# Use full dataset each round → more accurate but overfit risk

subsample=0.8
# Use 80% of data → slightly noisier trees → more generalizable
```

---

### 🔷 5. `colsample_bytree` — *What % of features (columns) to use*

#### 🔍 Intuition:

* Randomly selects a subset of features per tree → improves robustness

#### 🛠 Practical Use:

* Try values from `0.5 to 1.0`

```python
colsample_bytree = 1.0
# Use all features each time

colsample_bytree = 0.7
# Use 70% of features → prevents model from relying too much on 1 or 2 features
```

---

### 🔷 6. `gamma` — *Minimum loss reduction to make a split*

#### 🔍 Intuition:

* Controls tree **conservativeness**
* Higher gamma = model only makes splits that reduce error significantly

#### 🛠 Practical Use:

* Good for pruning
* Try values: `0, 1, 5, 10`

```python
gamma = 0
# Split freely to reduce error

gamma = 5
# Only split if it improves the objective a lot → prevents overfitting
```

---

### 🔷 7. `lambda` and `alpha` — *Regularization (like in Ridge/Lasso)*

| Parameter | Effect                       | Use When                  |
| --------- | ---------------------------- | ------------------------- |
| `lambda`  | L2 regularization on weights | Helps with collinearity   |
| `alpha`   | L1 regularization on weights | Can drive weights to zero |

#### 🛠 Practical Use:

* Use these if your model is **overfitting** or **too complex**

---

### 🔷 8. `objective` — *Defines the problem type*

| Value              | Use Case                  |
| ------------------ | ------------------------- |
| `multi:softmax`    | Multiclass classification |
| `binary:logistic`  | Binary classification     |
| `reg:squarederror` | Regression                |

---

### 🧪 Tuning Strategy:

1. **Start with default values**:

```python
params = {
  'objective': 'binary:logistic',
  'max_depth': 6,
  'eta': 0.3,
  'subsample': 1,
  'colsample_bytree': 1,
}
```

2. **If overfitting**:

   * Lower `max_depth`
   * Add `gamma`
   * Lower `eta`
   * Add regularization: `lambda` and `alpha`

3. **If underfitting**:

   * Increase `max_depth`
   * Reduce regularization
   * Increase `num_boost_round`

---

