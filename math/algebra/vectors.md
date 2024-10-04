## Vectors

Below are the questions and answers of questions related to vectors.

### Dot Product
#### What's the geometric interpretation of the dot product of two vectors?

The dot product of two vectors $\mathbf{a}$ and $\mathbf{b}$ in Euclidean space can be interpreted geometrically as:

$$
\mathbf{a} \cdot \mathbf{b} = \mathbf{\vert{a}\vert} \mathbf{\vert{b}\vert} \cos(\theta)
$$

where:
- $\mathbf{\vert{a}\vert}$ and $\mathbf{\vert{b}\vert}$ are the magnitudes (lengths) of the vectors, and
- $\mathbf{\theta}$ is the angle between the two vectors.

##### Key Geometrical Insights:
1. **Cosine Relationship**: The dot product encodes the cosine of the angle between the vectors. If the vectors point in the same direction, $\mathbf{\cos(\theta) = 1}$, and the dot product is maximized. If they are orthogonal (at 90°), $\mathbf{\cos(\theta) = 0}$, and the dot product is zero. If they point in opposite directions, $\mathbf{\cos(\theta) = -1}$, and the dot product is negative.
   
2. **Projection**: The dot product can also be interpreted as the magnitude of one vector projected onto the other. Specifically, $\mathbf{a \cdot b = \vert{a}\vert \vert{b}\vert \cos(\theta)}$ is the magnitude of $\mathbf{a}$ in the direction of $\mathbf{b}$, scaled by $\mathbf{\vert{b}\vert}$.

3. **Parallelism and Orthogonality**: When the dot product is zero, the vectors are orthogonal (perpendicular), indicating no alignment. When the dot product is positive, the vectors form an acute angle; when negative, they form an obtuse angle.

##### Practical Significance in Machine Learning:
In machine learning and deep learning, the dot product frequently shows up in operations like calculating the similarity between vectors (e.g., in word embeddings), and in neural network layers where it's used for computing linear transformations.

#### Given a vector $\mathbf{u}$, find vector $\mathbf{v}$ of unit length such thsat the dot product of $\mathbf{u}$ and $\mathbf{v}$ is maximum.

To maximize the dot product between two vectors $\mathbf{u}$ and $\mathbf{v}$, we can leverage the geometric interpretation of the dot product:

$$
u \cdot v = \vert{u}\vert \cos(\theta)
$$

where:

- $\mathbf{\vert{u}\vert}$ and $\mathbf{\vert{v}\vert}$ are the magnitudes (lengths) of the vectors, and
- $\mathbf{\theta}$ is the angle between the vectors.

Since we are looking for v to have a unit length (i.e., |v| = 1), the equation simplifies to:

$$
v = u / \vert{u}\vert
$$

This expression ensures that:
- $\mathbf{v}$ has unit length (i.e. $\mathbf{\vert{v}\vert = 1}$), and
- The dot product is maximized as $\mathbf{v}$ is aligned with $\mathbf{u}$.

##### Practical Significance:
In machine learning, this concept is useful in many applications, such as maximizing similarity between vectors (e.g., in embeddings) or optimizing linear models, where aligning a weight vector with the input maximizes the model’s response.

### Outer Product
#### Given two vectors $\mathbf{a = [3, 2, 1]}$ and $\mathbf{b = [-1, 0, 1]}$, calculate the outer product $\mathbf{a^T b}$.

The outer product of two vectors $\mathbf{a}$ and $\mathbf{b}$ results in a matrix. For vectors $\mathbf{a}$ (with shape $n \times 1$) and $\mathbf{b}$ (with shape $1 \times m$), the outer product is a matrix where each element $C_{ij}$ is computed as:

$$
C_{ij} = a_i * b_j
$$

For the vectors $\mathbf{a = [3, 2, 1]}$ and $\mathbf{b = [-1, 0, 1]}$, the outer product $\mathbf{a^T b}$ can be calculated as follows:

$$
a^Tb = \begin{bmatrix}
3 \cdot (-1)  &  3 \cdot 0  &  3 \cdot 1 \\
2 \cdot (-1)  &  2 \cdot 0  &  2 \cdot 1 \\
1 \cdot (-1)  &  1 \cdot 0  &  1 \cdot 1
\end{bmatrix}
$$

This gives us:

$$
a^T b = 
\begin{bmatrix}
-3 & 0 & 3 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
$$

##### Practical Significance:
The outer product is a fundamental operation in linear algebra and is widely used in machine learning for constructing matrices from vectors, often applied in tasks such as tensor operations and in neural networks.

#### Give an example of how the outer product can be useful in ML.

The outer product is a key operation in many machine learning applications. One practical example is in the construction of **covariance matrices**, which are essential in understanding the relationships between different features in a dataset.

##### Covariance Matrix Example:
Suppose you have a dataset with two feature vectors, $\mathbf{x}$ and $\mathbf{y}$, representing two different variables observed over several samples:

$$
x = [x_1, x_2, …, x_n]
$$

$$
y = [y_1, y_2, …, y_n]
$$

To understand how these two features relate to each other, you can compute their covariance. The covariance matrix is calculated as:

$$
cov(x, y) = E[(x - E[x])(y - E[y])^T]
$$

The outer product comes into play in the term $(x - E[x])(y - E[y])^T$, which computes the pairwise product between the deviations of the vectors $\mathbf{x}$ and $\mathbf{y}$ from their means. This outer product gives you insight into how these two features vary together. If the outer product results in large positive values, it means that $\mathbf{x}$ and $\mathbf{y}$ tend to increase together; if it’s negative, one increases while the other decreases.

##### Tensor Factorization in Neural Networks:

Another important application of the outer product in machine learning is in **tensor factorization**. In some deep learning architectures (such as attention mechanisms or in recommendation systems), interactions between features are often modeled using outer products. By taking the outer product of two vectors, you capture all pairwise interactions between their components, which can be used for learning complex relationships between features.

##### Word Embeddings:

In natural language processing (NLP), the outer product can be used in word embedding techniques like **word2vec** to build a co-occurrence matrix. The co-occurrence matrix is created by calculating the outer product of word vectors based on their context in sentences. This matrix helps in understanding how frequently certain words appear together and can be used to capture semantic relationships between words.

### What does it mean for two vectors to be linearly independent?

Two vectors $\mathbf{a}$ and $\mathbf{b}$ are said to be **linearly independent** if no scalar multiple of one vector can be used to express the other. In other words, there is no scalar $\mathbf{c}$ such that:

$$
a = c \cdot b
$$

or equivalently:

$$
b = c \cdot a
$$

If such a scalar exists, the vectors are **linearly dependent**. If not, they are **linearly independent**.

##### Geometric Interpretation:
- **Linearly independent vectors** point in different directions. In the case of two vectors in a 2D plane, this means they span the plane, and any vector in that plane can be represented as a linear combination of the two vectors.
- If two vectors are **linearly dependent**, they lie on the same line or are parallel, meaning one is just a scaled version of the other.

##### Why Linear Independence is Important in Machine Learning:
1. **Feature Independence**: In machine learning, linear independence is crucial for feature selection. If features (represented as vectors) are linearly dependent, it means they carry redundant information, which can negatively impact the model’s performance by leading to multicollinearity. Models such as linear regression assume that features are linearly independent to avoid redundancy.

2. **Basis for Vector Spaces**: Linear independence is fundamental for constructing a **basis** for a vector space. In data science, the concept of a basis helps in representing data in lower-dimensional spaces, such as in **Principal Component Analysis (PCA)**, where linearly independent components are used to capture the most variance in data.

##### Example:
Consider two vectors in 2D space:

$$
a = [1, 2] \quad \text{and} \quad b = [3, 6]
$$

In this case, $\mathbf{b}$ is just a scalar multiple of $\mathbf{a}$ ($\mathbf{b = 3 \cdot a}$). Hence, $\mathbf{a}$ and $\mathbf{b}$ are linearly dependent.

On the other hand, if:

$$
a = [1, 2] \quad \text{and} \quad b = [2, 3]
$$

then no scalar multiple of $\mathbf{a}$ can give $\mathbf{b}$, so $\mathbf{a}$ and $\mathbf{b}$ are linearly independent.

### Given two sets of vectors $\mathbf{A = {a_1, a_2, a_2, ... , a_n}}$ and $\mathbf{B = {b_1, b_2, ..., b_m}}$, how do you check that they share the same basis?

To determine if two sets of vectors $\mathbf{A}$ and $\mathbf{B}$ share the same basis, you need to verify that both sets span the same vector space. This requires the following steps:

1. **Check that the sets span the same subspace**:
   - Combine all the vectors from both sets into one matrix and reduce it to row echelon form (or reduced row echelon form). 
   - If the number of linearly independent vectors is the same for both sets, and they span the same space, then $\mathbf{A}$ and $\mathbf{B}$ share the same basis.
   
2. **Linearly combine vectors from one set using the other**:
   - For each vector in $\mathbf{A}$, check if it can be written as a linear combination of vectors in $\mathbf{B}$. Similarly, check if each vector in $\mathbf{B}$ can be written as a linear combination of vectors in $\mathbf{A}$.
   - If all vectors in one set can be written as linear combinations of vectors in the other set, the sets share the same basis.

##### Detailed Steps:

1. **Form matrices from both sets**:
   - Construct matrix $\mathbf{M_A}$ by placing vectors from set $\mathbf{A}$ as columns: 
   
   $M_A = [a_1, a_2, ..., a_n]$

   - Similarly, construct matrix $\mathbf{M_B}$ from vectors of set $\mathbf{B}$: 
   
   $M_B = [b_1, b_2, ..., b_m]$

2. **Perform rank analysis**:
   - Calculate the rank (number of linearly independent vectors) of both matrices, $\mathbf{rank(M_A)}$ and $\mathbf{rank(M_B)}$.
   - If $\mathbf{rank(M_A)}$ = $\mathbf{rank(M_B)}$, the sets may share the same basis.

3. **Test if one set is a linear combination of the other**:
   - Solve the system of equations to check if each vector in $\mathbf{A}$ can be expressed as a linear combination of vectors in $\mathbf{B}$, and vice versa.
   - To test this, for each $\mathbf{a_i}$, solve the equation:
   
   $a_i = c_1 \cdot b_1 + c_2 \cdot b_2 + ... + c_m \cdot b_m$

   If a solution exists for every $\mathbf{a_i}$ and for every $\mathbf{b_i}$, then the sets share the same basis.

##### Example:

Consider the sets of vectors $\mathbf{A}$ and $\mathbf{B}$:
- $A = { [1, 0], [0, 1] }$
- $B = { [1, 1], [1, -1] }$

Step 1: Create matrices from the vectors:

$$
M_A = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}, \quad
M_B = \begin{bmatrix}
1 & 1 \\
1 & -1 
\end{bmatrix}
$$

Step 2: Find the rank of both matrices. Both $\mathbf{M_A}$ and $\mathbf{M_B}$ have rank 2, indicating that both sets are linearly independent.

Step 3: Test linear combinations. We can write each vector in $\mathbf{A}$ as a combination of vectors in $\mathbf{B}$:
- $[1, 0] = 0.5 \cdot [1, 1] + 0.5 \cdot [1, -1]$
- $[0, 1] = 0.5 \cdot [1, 1] - 0.5 \cdot [1, -1]$

Since each vector in $\mathbf{A}$ can be expressed as a linear combination of vectors in $\mathbf{B}$, and vice versa, $\mathbf{A}$ and $\mathbf{B}$ share the same basis.

##### Practical Significance:
In machine learning and data science, identifying if two sets of features share the same basis is crucial in feature engineering and dimensionality reduction techniques such as **Principal Component Analysis (PCA)**. It helps ensure that the new feature set spans the same space as the original, preserving the essential information.

### Given $\mathbf{n}$ vectors, each of $\mathbf{d}$ dimensions. What is the dimension of their span?

The dimension of the **span** of a set of vectors refers to the number of linearly independent vectors in the set. The span of a set of vectors is the vector space formed by all possible linear combinations of those vectors. The dimension of this vector space depends on the number of linearly independent vectors in the set.

##### Key points to determine the dimension of the span:

1. **Maximum dimension**: The dimension of the span can never be greater than the number of dimensions of the vectors, which is $\mathbf{d}$. So, if you have $\mathbf{n}$ vectors in $\mathbf{d}$-dimensional space, the dimension of their span is at most $\mathbf{min(n, d)}$.

2. **Linear independence**: 
   - If all $\mathbf{n}$ vectors are **linearly independent**, the dimension of the span will be $\mathbf{min(n, d)}$.
   - If the vectors are **linearly dependent**, the dimension of the span will be less than $\mathbf{n}$, depending on how many vectors are linearly independent.

##### Case 1: $\mathbf{n\leq d}$
If the number of vectors $\mathbf{n}$ is less than or equal to the dimensionality $\mathbf{d}$, the dimension of the span will be equal to the number of linearly independent vectors in the set. In the best-case scenario, where all vectors are linearly independent, the dimension of the span will be exactly $\mathbf{n}$.

Example:  
Consider $\mathbf{n = 3}$ vectors in $\mathbf{d = 5}$ dimensional space. If the vectors are linearly independent, the dimension of their span is $\mathbf{3}$.

##### Case 2: $\mathbf{n\gt d}$
When the number of vectors $\mathbf{n}$ exceeds the dimensionality $\mathbf{d}$, it is impossible for all vectors to be linearly independent. In this case, the maximum number of linearly independent vectors can only be $\mathbf{d}$. So the dimension of the span is capped at $\mathbf{d}$.

Example:  
Consider $\mathbf{n = 6}$ vectors in $\mathbf{d = 3}$ dimensional space. Since no more than 3 vectors can be linearly independent in a 3-dimensional space, the dimension of their span will be $\mathbf{3}$.

##### General Rule:
- The dimension of the span is the number of linearly independent vectors, which is $\mathbf{min(n, d)}$.

##### Practical Significance:
In machine learning, the dimension of the span of feature vectors is crucial for understanding the effective dimensionality of the data. For example, in **Principal Component Analysis (PCA)**, the principal components represent the directions of maximum variance, which correspond to the independent directions (or span) in the dataset. The span of these principal components is often much lower than the original dimensionality, which helps in dimensionality reduction.

### Norm and metrics
#### What's a norm? What is $\mathbf{L_0}$, $\mathbf{L_1}$, $\mathbf{L_2}$, and $\mathbf{L_norm}$?

A **norm** is a function that assigns a non-negative length or size to a vector in a vector space. In simple terms, it measures the "magnitude" of a vector. Norms are widely used in various fields such as mathematics, physics, and machine learning to quantify the size of vectors.

Norms have several important properties:
1. **Non-negativity**: $\vert{x}\vert\geq 0$
2. **Definiteness**: $\vert{x}\vert\= 0$ if and only if $x = 0$
3. **Homogeneity**: $\vert{\alpha x}\vert = \vert{\alpha}\vert \cdot \vert{x}\vert$ (scaling property)
4. **Triangle inequality**: $\vert{x + y}\vert\leq \vert{x}\vert + \vert{y}\vert$

Now, let's discuss specific types of norms used in machine learning:

##### $\mathbf{L_0}$ norm:
The $\mathbf{L_0}$ **norm** counts the number of **non-zero elements** in the vector. It doesn't technically satisfy all the properties of a norm (specifically, the triangle inequality), but it is often referred to as a "norm" in practice.

Formula: $\vert{x}\vert = number of non zero elements in x$

Example: For a vector $x = [3, 0, 4, 0]$, $\vert{x}\vert = 2$ (because there are two non-zero elements: 3 and 4)

**Use in Machine Learning**:  
The L₀ norm is often used in **sparse learning** methods, where the goal is to minimize the number of non-zero features in a model. However, due to its discrete nature, the L₀ norm is non-differentiable and computationally expensive.

##### **L₁ Norm (ℓ₁ norm)**:
- The **L₁ norm** is the sum of the **absolute values** of the vector elements. It measures the "taxicab" distance between points.
- Formula:
  \[
  \|x\|₁ = \sum_{i=1}^{n} |x_i|
  \]
- Example:  
  For a vector \( x = [3, -4, 5] \),  
  \( \|x\|₁ = |3| + |-4| + |5| = 12 \).

**Use in Machine Learning**:  
The L₁ norm is used in **Lasso regression** for feature selection, where it encourages sparsity in the model coefficients, setting many of them to zero.

##### **L₂ Norm (ℓ₂ norm or Euclidean norm)**:
- The **L₂ norm** is the **square root of the sum of the squares** of the vector elements. This is the most commonly used norm and represents the Euclidean distance from the origin to the vector.
- Formula:
  \[
  \|x\|₂ = \sqrt{\sum_{i=1}^{n} x_i^2}
  \]
- Example:  
  For a vector \( x = [3, -4, 5] \),  
  \( \|x\|₂ = \sqrt{3^2 + (-4)^2 + 5^2} = \sqrt{9 + 16 + 25} = \sqrt{50} \approx 7.07 \).

**Use in Machine Learning**:  
The L₂ norm is used in **Ridge regression** and **SVMs** to minimize the magnitude of the model coefficients, preventing overfitting.

##### **L_p Norm**:
- The **L_p norm** generalizes the L₁ and L₂ norms and is defined for any positive real number **p**.
- Formula:
  \[
  \|x\|_p = \left