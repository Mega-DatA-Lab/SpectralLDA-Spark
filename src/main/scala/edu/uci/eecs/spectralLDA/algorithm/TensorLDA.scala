package edu.uci.eecs.spectralLDA.algorithm

/**
 * Tensor Decomposition Algorithms.
 * Alternating Least Square algorithm is implemented.
 * Created by Furong Huang on 11/2/15.
 */
import edu.uci.eecs.spectralLDA.datamoments.DataCumulant
import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, argsort, diag, norm, *}
import breeze.numerics._
import breeze.stats.distributions.{Rand, RandBasis}
import edu.uci.eecs.spectralLDA.utils.L1SimplexProjection
import org.apache.spark.rdd.RDD


/** Spectral LDA model
  *
  * @param dimK                       number of topics k
  * @param alpha0                     sum of alpha for the Dirichlet prior
  * @param maxIterations              max number of iterations for the ALS,
  *                                   500 by default
  * @param tol                        tolerance for checking convergence
  *                                   of ALS, 1e-6 by default
  * @param randomisedSVD              uses randomised SVD on M2,
  *                                   true by default
  * @param numIterationsKrylovMethod  iterations of the Krylov Method for
  *                                   randomised SVD, 2 by default
  * @param postProcessing             post-processing topic-word distribution
  *                                   matrix by projection into l1-simplex,
  *                                   false by default
  */
class TensorLDA(dimK: Int,
                alpha0: Double,
                maxIterations: Int = 500,
                tol: Double = 1e-6,
                randomisedSVD: Boolean = true,
                numIterationsKrylovMethod: Int = 2,
                postProcessing: Boolean = false) extends Serializable {
  assert(dimK > 0, "The number of topics dimK must be positive.")
  assert(alpha0 > 0, "The topic concentration alpha0 must be positive.")
  assert(maxIterations > 0, "The number of iterations for ALS must be positive.")
  assert(tol > 0.0, "tol must be positive and probably close to 0.")

  def fit(documents: RDD[(Long, SparseVector[Double])])
         (implicit randBasis: RandBasis = Rand)
          : (DenseMatrix[Double], DenseVector[Double],
             DenseMatrix[Double], DenseVector[Double],
             DenseVector[Double]) = {
    val cumulant: DataCumulant = DataCumulant.getDataCumulant(
      dimK,
      alpha0,
      documents,
      randomisedSVD = randomisedSVD,
      numIterationsKrylovMethod = numIterationsKrylovMethod
    )

    val myALS: ALS = new ALS(
      dimK,
      cumulant.thirdOrderMoments,
      maxIterations = maxIterations,
      tol = tol
    )

    val (nu1: DenseMatrix[Double], nu2: DenseMatrix[Double], nu3: DenseMatrix[Double],
      lambda: DenseVector[Double]) = myALS.run

    val nu = uniqueFactor(nu1, nu2, nu3)

    // unwhiten the results
    // unwhitening matrix: $(W^T)^{-1}=U\Sigma^{1/2}$
    val unwhiteningMatrix = cumulant.eigenVectorsM2 * diag(sqrt(cumulant.eigenValuesM2))

    val alphaUnordered: DenseVector[Double] = lambda.map(x => scala.math.pow(x, -2))
    val topicWordMatrixUnordered: DenseMatrix[Double] = unwhiteningMatrix * nu * diag(lambda)

    // re-arrange alpha and topicWordMatrix in descending order of alpha
    val idx = argsort(alphaUnordered).reverse.take(dimK)
    val alpha = alphaUnordered(idx).toDenseVector
    val topicWordMatrix = topicWordMatrixUnordered(::, idx).toDenseMatrix

    if (postProcessing)
      (topicWordMatrix(::, *).map(L1SimplexProjection.project), alpha,
        cumulant.eigenVectorsM2, cumulant.eigenValuesM2,
        cumulant.firstOrderMoments)
    else
      (topicWordMatrix, alpha,
       cumulant.eigenVectorsM2, cumulant.eigenValuesM2,
       cumulant.firstOrderMoments)
  }

  private def uniqueFactor(nu1: DenseMatrix[Double],
                           nu2: DenseMatrix[Double],
                           nu3: DenseMatrix[Double]): DenseMatrix[Double] = {
    val nu = nu1.copy
    val eps = 1e-12
    for (j <- 0 until nu1.cols) {
      val diff1 = norm(nu2(::, j) - nu3(::, j))
      val diff2 = norm(nu1(::, j) - nu3(::, j))
      val diff3 = norm(nu1(::, j) - nu2(::, j))

      if (diff1 < diff2 + eps && diff1 < diff3 + eps)
        nu(::, j) := nu1(::, j)
      else if (diff2 < diff1 + eps && diff2 < diff3 + eps)
        nu(::, j) := nu2(::, j)
      else if (diff3 < diff1 + eps && diff3 < diff2 + eps)
        nu(::, j) := nu3(::, j)
    }

    nu
  }
}
