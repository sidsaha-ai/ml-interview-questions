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
