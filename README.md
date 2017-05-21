# Spectral LDA on Spark

## Summary 
This code implements a Spectral (third order tensor decomposition) learning method for learning LDA topic model on Spark.

## How do I get set up?
We use the `sbt` build system. By default we support Scala 2.11.8 and Spark 2.0.0 upward. Just run

```bash
sbt package
```

It will produce `target/scala-2.11/spectrallda-tensor_2.11-1.1.jar`.
    
### Command Line Usage
The command line usage is 
    
```bash
Spectral LDA Factorization
Usage: SpectralLDA [options] <input>...

  -k, --k <value>          number of topics
  --alpha0 <value>         sum of the topic distribution prior parameter
  --max-iter <value>       number of iterations of learning. default: 500
  --tol <value>            tolerance for the ALS algorithm. default: 1.0E-6
  -o, --output-dir <dir>   output write path. default: .
  --help                   prints this usage text
  <input>...               paths of input files   
```
The higher `alpha0` is relative to `k` the more likely are we to recover only topic-specific words (vs "common" words that would exist in every topic distribution). If `alpha0 = k` we would allow a non-informative prior for the topic distribution, when every `alpha_i = 1.0`.

Currently `input-file` could only be a Hadoop SequenceFiles storing serialised `RDD[(Long, breeze.linalg.SparseVector[Double])]`.

An example call from command line is

```bash
spark-submit \
--packages com.github.scopt:scopt_2.11:3.5.0,org.apache.hadoop:hadoop-aws:2.7.3 \
--class edu.uci.eecs.spectralLDA.SpectralLDA \
target/scala-2.11/spectrallda-tensor_2.11-1.1.jar \
-k 5 --alpha0 5.0 -o results <RDD-file>
```

It runs with `alpha0 = k = 5` and outputs results in `result/`.

### API usage
The API is designed following the lines of the Spark built-in `LDA` class.

```scala
import edu.uci.eecs.spectralLDA.algorithm.TensorLDA
import breeze.linalg._

val lda = new TensorLDA(
  dimK = params.k,
  alpha0 = params.topicConcentration,
  maxIterations = value,            // optional, default: 500
  tol = value,                      // optional, default: 1e-6
  randomisedSVD = true,             // optional, default: true
  numIterationsKrylovMethod = value // optional, default: 1
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
import edu.uci.eecs.spectralLDA.algorithm.ALS
import breeze.linalg._

val als = new ALS(
  dimK = value,
  tensor3D = value,        // k-by-(k*k) matrix for the unfolded 3rd-order symmetric tensor
  maxIterations = value,            // optional, default: 500
  tol = value,                      // optional, default: 1e-6
)

// We run ALS to find the best approximating sum of rank-1 tensors such that 
// $$ M3 = \sum_{i=1}^k\alpha_i\beta_i^{\otimes 3} $$

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
