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
