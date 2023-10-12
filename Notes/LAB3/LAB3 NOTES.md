Logistic regression är ett sätt att klassificera. Där vi använder en logistisk funktion, i vårat fall 

**Sigmoid**:  $$\frac{1}{1+e^{-z}}$$
där z = är en linjär kombination av Egenskaper och vikterna: 

$$z = w0 * x0 + w1 * x1 + w2 * x2 + ... + wn * xn
$$
Hypotesen blir då: 
$$h(x) = sigmoid(z) = \frac{1}{1+e^{-z}}
$$
**Vikter(W):** Är en parameter som anpassas under träningsprocesses för att göra korrekta förutsägelser

**Egenskaper(x)**: Längd på en blomma exempelvis..

**Hypotes**: Vårat sannolikhetsvärde t.ex vara att vi får 0,98 från sigmoid, detta betyder att det är 98% chans att det är sant att det är en "Setosa" tex, vilket är en av klasserna i datan.



**Learning rate : 
1. A high learning rate may cause the optimization algorithm to overshoot the optimal parameters, leading to instability and failure to converge.
2. A low learning rate may cause slow convergence, requiring a large number of iterations to reach the optimal parameters.

**Epochs**:
1. Multiple epochs are often required to train a model effectively. The number of epochs is a hyperparameter that determines how many times the entire dataset is used to update the model parameters.
2. Too few epochs may result in an underfit model that hasn't learned the patterns in the data well.
3. Too many epochs may result in overfitting, where the model learns the training data too well but doesn't generalize well to unseen data.

Vi får bästa värdet vid 0.01 learing rate och 1000 epos