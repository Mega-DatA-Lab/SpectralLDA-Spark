# Spectral LDA on Spark

## Summary 
This code implements a Spectral (third order tensor decomposition) learning method for learning LDA topic model on Spark.

The Spectral learning method works with empirical counts of word pair or word triplet that appear in the same document. We collect and average them across documents. If we denote these empirical moments as tensors, we could orthogonalise and then perform the CANDECOMP/PARAFAC Decomposition on the 3rd-order moment tensor to recover the topic-word distributions of the LDA model. For more details, please refer to `report.pdf` in this repository.

## How do I get set up?
The code is written for Java 8, Scala 2.11.12, and Spark 2.3.0+. We use the `sbt` build system. Download the latest version of `sbt` and run

```bash
sbt package test
```

which will produce `target/scala-2.11/spectrallda-tensor_2.11-<version>.jar`. The version number is defined in `build.sbt`.

### Publish the Package
In order to use the classes within this project in an interactive shell or programmatically, run ```sbt publishLocal``` to publish the package to the local repository. Either invoke the Spark shell with option `--packages megadata:spectrallda-tensor_2.11:<version>` or add the following lines in the `<dependencies>` section of the `pom.xml` in your new project.

```xml
<dependency>
      <groupId>megadata</groupId>
      <artifactId>spectrallda-tensor_2.11</artifactId>
      <version>x.xx.xx</version>
</dependency>
```

### API Usage
The API is designed following the lines of the Spark built-in `LDA` class.

```scala
import megadata.spectralLDA.algorithm.TensorLDA

val lda = new TensorLDA(
  dimK = params.k,
  alpha0 = params.topicConcentration,
  maxIterations = value,             // optional, default: 500
  tol = value,                       // optional, default: 1e-6
  randomisedSVD = true,              // optional, default: true
  numIterationsKrylovMethod = value, // optional, default: 2
  postProcessing = value             // optional, default: false
)

// Fit against the documents
// beta is the V-by-k matrix, where V is the vocabulary size, 
// k is the number of topics. Each column stores the word distribution per topic
// alpha is the length-k Dirichlet prior parameter for the topic distribution

// eigvecM2 is the V-by-k matrix for the top k eigenvectors of M2
// eigvalM2 is the length-k vector for the top k eigenvalues of M2
// m1 is the length-V vector for the average word distribution

val (beta: DenseMatrix[Double], alpha: DenseVector[Double], 
  eigvecM2: DenseMatrix[Double], eigvalM2: DenseVector[Double],
  m1: DenseVector[Double]) = lda.fit(documents)
```

If one just wants to decompose a 3rd-order symmetric tensor into the sum of rank-1 tensors, we could do

```scala
import megadata.spectralLDA.algorithm.ALS

val als = new ALS(
  dimK = value,
  tensor3D = value,        // k-by-(k*k) matrix for the unfolded 3rd-order symmetric tensor
  maxIterations = value,            // optional, default: 500
  tol = value,                      // optional, default: 1e-6
)

// We run ALS to find the best approximating sum of rank-1 tensors such that 
// <math> M3 = \sum_{i=1}^k\alpha_i\beta_i^{\otimes 3} </math>

// beta are the factor matrices
// alpha is the eigenvalue vector
val (beta1, beta2, beta3, alpha: DenseVector[Double]) = als.run
```
    
## References
* White Paper: http://newport.eecs.uci.edu/anandkumar/pubs/whitepaper.pdf
* New York Times Result Visualization: http://newport.eecs.uci.edu/anandkumar/Lab/Lab_sub/NewYorkTimes3.html

## Who do I talk to?

* Repo owner or admin: Furong Huang 
* Contact: furongh.uci@gmail.com
