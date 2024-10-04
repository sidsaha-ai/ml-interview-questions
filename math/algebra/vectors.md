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
u \cdot v = \vert{u}\vert \vert{b}\ \cos(\theta)
$$

where:

- $\mathbf{\vert{u}\vert}$ and $\mathbf{\vert{v}\vert}$ are the magnitudes (lengths) of the vectors, and
- $\mathbf{\theta}$ is the angle between the vectors.

Since we are looking for v to have a unit length (i.e., |v| = 1), the equation simplifies to:

$$
v = u / \vert{u}\vert
$$

This expression ensures that:
- $\mathb{v}$ has unit length (i.e. $\mathbf{\vert{v}\vert = 1}$), and
- The dot product is maximized as $\mathbf{v}$ is aligned with $\mathbf{u}$.

##### Practical Significance:
In machine learning, this concept is useful in many applications, such as maximizing similarity between vectors (e.g., in embeddings) or optimizing linear models, where aligning a weight vector with the input maximizes the model’s response.