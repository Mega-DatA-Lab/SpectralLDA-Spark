package megadata.spectralLDA.algorithm

import breeze.linalg.eigSym.EigSym
import breeze.linalg.qr.QR
import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, argtopk, diag, eigSym, qr}
import breeze.numerics.sqrt
import breeze.stats.distributions.{Gaussian, Rand, RandBasis}
import org.apache.spark.rdd.RDD

/** Performs Randomised Numerical Linear Algebra */
object RandNLA {
  /** Randomised Power Iteration Method for SVD of shifted M2
    *
    * As the shifted M2 is of size V-by-V, where V is the vocabulary size, which could be
    * very large, we carry out Randomised Power Iteration Method for computing the SVD of it
    * following Musco & Musco 2016. The Nystrom decomposition is also implemented but it
    * gives much worse empirical performance than Musco & Musco 2016.
    *
    * Ref:
    * Halko, N, P.G. Martinsson, & J.A. Tropp, Finding Structure with Randomness:
    * Probabilistic Algorithms for Constructing Approximate Matrix Decompositions, 2011
    * Gu, Ming, Subspace Iteration Randomization and Singular Value Problems, 2014
    * Musco, Cameron, & Christopher Musco, Randomized Block Krylov Methods for Stronger
    * and Faster Approximate Singular Value Decomposition, 2016
    *
    * @param alpha0     sum of alpha, the Dirichlet prior vector
    * @param vocabSize  V: the vocabulary size
    * @param dimK       K: number of topics
    * @param numDocs    number of documents
    * @param firstOrderMoments   average of the word count vectors
    * @param documents  RDD of the documents
    * @param nIter      number of iterations for the Randomised Power Iteration method,
    *                   denoted by q in the Algorithm 1 & 2 in the ref paper
    * @param randBasis  the random seed
    * @return           V-by-K eigenvector matrix and length-K eigenvalue vector
    */
  def whiten2(alpha0: Double,
              vocabSize: Int,
              dimK: Int,
              numDocs: Long,
              firstOrderMoments: DenseVector[Double],
              documents: RDD[(Long, Double, SparseVector[Double])],
              nIter: Int = 1)
            (implicit randBasis: RandBasis = Rand)
  : (DenseMatrix[Double], DenseVector[Double]) = {
    assert(vocabSize >= dimK)
    assert(nIter >= 0)

    // The following paper discusses about the success of random
    // projection with relation to the dimension of the sub-space
    // we're concerned about. i.e. To get the first dimK eigenvectors
    // we better project into (dimK + certain slack) sub-space.
    //
    // Ref:
    // Universality laws for randomized dimension reduction
    // with applications, S. Oymak and J. A. Tropp. Inform. Inference, Nov. 2017
    // Theorem II on Restricted Minimum Singular Value
    val projectedDim = math.pow(dimK, 1.1).toInt

    // Cache some data
    val extDocs = documents
      .map {
        case (docId, len, v) => (docId, len, v, 1.0 / len / (len - 1))
      }
    val normalisedDocs = extDocs
      .map {
        case (_, _, v, c2) => v * c2
      }
      .reduce(_ + _)
      .toDenseVector

    // Random test matrix
    var q = DenseMatrix.rand[Double](vocabSize, projectedDim,
      Gaussian(mu = 0.0, sigma = 1.0))

    // Product of M2 with the test matrix
    var m2q: DenseMatrix[Double] = null
    val tmpResult = DenseMatrix.zeros[Double](vocabSize, projectedDim)

    for (i <- 0 until 2 * nIter + 1) {
      m2q = randomProjectM2(
        extDocs,
        q,
        alpha0,
        numDocs,
        firstOrderMoments,
        normalisedDocs,
        tmpResult
      )
      val QR(nextq, _) = qr.reduced(m2q)
      q = nextq
    }

    m2q = randomProjectM2(
      extDocs,
      q,
      alpha0,
      numDocs,
      firstOrderMoments,
      normalisedDocs,
      tmpResult
    )

    // Only take the top dimK eigenvalues
    val (s: DenseVector[Double], u: DenseMatrix[Double]) = decomp2(m2q, q)
    val idx = argtopk(s, dimK)
    (u(::, idx).toDenseMatrix, s(idx).toDenseVector)
  }

  /** Musco-Musco method for randomised eigendecomposition of Hermitian matrix
    *
    * We could first do the eigendecomposition (AQ)^* AQ=USU^*. If A=HKH^*; apparently
    * K=S^{1/2}, H=QU.
    *
    * Empirically for the Spectral LDA model, this decomposition algorithm often
    * gives better final results than the Nystrom method.
    *
    * Ref:
    * Musco, Cameron, & Christopher Musco, Randomized Block Krylov Methods for Stronger
    * and Faster Approximate Singular Value Decomposition, 2016
    *
    * @param aq product of the original n-by-n matrix A and a n-by-k test matrix
    * @param q  the n-by-k test matrix
    * @return   the top k eigenvalues, top k eigenvectors of the original matrix A
    */
  def decomp2(aq: DenseMatrix[Double],
              q: DenseMatrix[Double])
      : (DenseVector[Double], DenseMatrix[Double]) = {
    val w = aq.t * aq
    val EigSym(s: DenseVector[Double], u: DenseMatrix[Double]) = eigSym((w + w.t) / 2.0)

    val sqrt_s = sqrt(s)

    val h = q * u

    (sqrt_s, h)
  }

  /** Given a test matrix q returns the product of shifted M2 and q */
  private[algorithm] def randomProjectM2(documents: RDD[(Long, Double,
                                            SparseVector[Double], Double)],
                                         q: DenseMatrix[Double],
                                         alpha0: Double,
                                         numDocs: Long,
                                         firstOrderMoments: DenseVector[Double],
                                         normalisedDocs: DenseVector[Double],
                                         tmpResult: DenseMatrix[Double]
                                         ): DenseMatrix[Double] = {
    val para_main: Double = (alpha0 + 1.0) * alpha0
    val para_shift: Double = alpha0 * alpha0

    tmpResult := 0.0
    documents
      .flatMap {
        case (_, _, v, c2) =>
          val proj = (q.t * v).toDenseVector * c2
          v.activeIterator.map {
            case (wid, cnt) => (wid, proj * cnt)
          }
      }
      .reduceByKey(_ + _)
      .collect
      .foreach {
        case (wid, y) => tmpResult(wid, ::) := y.t
      }

    val projectedWordPairsMatrix = (tmpResult
       - diag(normalisedDocs) * q) * (1.0 / numDocs)

    val projectedShiftedM2 = (projectedWordPairsMatrix * para_main
      - (firstOrderMoments * (firstOrderMoments.t * q)) * para_shift)

    projectedShiftedM2
  }
}
