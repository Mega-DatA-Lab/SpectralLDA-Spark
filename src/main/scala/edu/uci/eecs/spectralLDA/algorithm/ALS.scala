package edu.uci.eecs.spectralLDA.algorithm

/**
* Tensor Decomposition Algorithms.
* Alternating Least Square algorithm is implemented.
* Created by Furong Huang on 11/2/15.
*/

import edu.uci.eecs.spectralLDA.utils.Tensors
import breeze.linalg.{*, DenseMatrix, DenseVector, all, diag, max, min, norm, svd}
import breeze.stats.distributions.{Gaussian, Rand, RandBasis}
import scalaxy.loops._
import scala.language.postfixOps
import org.apache.log4j.Logger

/** CANDECOMP/PARAFAC Decomposition via Alternating Least Square (ALS)
  *
  * The current implementation only works for symmetric 3D tensors.
  *
  * @param dimK               tensor T is of shape dimK-by-dimK-by-dimK
  * @param tensor3D           dimK-by-(dimK*dimK) matrix for the unfolded M3
  * @param maxIterations      max iterations for the ALS algorithm
  * @param tol                tolerance. the dot product threshold is 1-tol
  * @param restarts           number of restarts of the ALS loop
  */
class ALS(dimK: Int,
          tensor3D: DenseMatrix[Double],
          maxIterations: Int = 500,
          tol: Double = 1e-6,
          restarts: Int = 5)
  extends Serializable {
  assert(dimK > 0, "The number of topics dimK must be positive.")
  assert(tensor3D.rows == dimK && tensor3D.cols == dimK * dimK,
    "The tensor3D must be dimK-by-(dimK * dimK) unfolded matrix")

  assert(maxIterations > 0, "Max iterations must be positive.")
  assert(tol > 0.0, "tol must be positive and probably close to 0.")
  assert(restarts > 0, "Number of restarts for ALS must be positive.")

  @transient private lazy val logger = Logger.getLogger("ALS")

  /** Run Alternating Least Squares (ALS)
    *
    * @param randBasis   default random seed
    * @return            three dimK-by-dimK matrices with
    *                    all the <math>beta_i</math> as columns,
    *                    length-dimK vector for all the eigenvalues
    */
  def run(implicit randBasis: RandBasis = Rand)
     : (DenseMatrix[Double], DenseMatrix[Double],
        DenseMatrix[Double], DenseVector[Double])={
    val gaussian = Gaussian(mu = 0.0, sigma = 1.0)

    var optimalA = DenseMatrix.zeros[Double](dimK, dimK)
    var optimalB = DenseMatrix.zeros[Double](dimK, dimK)
    var optimalC = DenseMatrix.zeros[Double](dimK, dimK)
    var optimalLambda = DenseVector.zeros[Double](dimK)

    var reconstructedLoss: Double = 0.0
    var optimalReconstructedLoss: Double = Double.PositiveInfinity

    val svd.SVD(a0, _, _) = svd(tensor3D)
    for (s <- 0 until restarts) {
      var A = a0.copy
      var B = a0.copy
      var C = a0.copy

      var A_prev = DenseMatrix.zeros[Double](dimK, dimK)
      var lambda: breeze.linalg.DenseVector[Double] = DenseVector.zeros[Double](dimK)

      logger.info("Start ALS iterations...")
      var iter: Int = 0
      while ((iter == 0) || (iter < maxIterations &&
        !isConverged(A_prev, A, dotProductThreshold = 1 - tol))) {
        A_prev = A.copy

        val (updatedA, updatedLambda1) = updateALSIteration(tensor3D, B, C)
        A = updatedA
        lambda = updatedLambda1

        val (updatedB, updatedLambda2) = updateALSIteration(tensor3D, C, A)
        B = updatedB
        lambda = updatedLambda2

        val (updatedC, updatedLambda3) = updateALSIteration(tensor3D, A, B)
        C = updatedC
        lambda = updatedLambda3

        logger.info(s"iter $iter\tlambda: max ${max(lambda)}, min ${min(lambda)}")

        iter += 1
      }
      logger.info("Finished ALS iterations.")

      reconstructedLoss = Tensors.dmatrixNorm(tensor3D - A * diag(lambda) * Tensors.krprod(C, B).t)
      logger.info(s"Reconstructed loss: $reconstructedLoss\tOptimal reconstructed loss: $optimalReconstructedLoss")

      if (reconstructedLoss < optimalReconstructedLoss) {
        optimalA = A
        optimalB = B
        optimalC = C
        optimalLambda = lambda
        optimalReconstructedLoss = reconstructedLoss
      }
    }

    (optimalA, optimalB, optimalC, optimalLambda)
  }

  private def updateALSIteration(unfoldedM3: DenseMatrix[Double],
                                 B: DenseMatrix[Double],
                                 C: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double]) = {
    val updatedA = unfoldedM3 * Tensors.krprod(C, B) * Tensors.to_invert(C, B)
    val lambda = norm(updatedA(::, *)).t.toDenseVector
    (matrixNormalization(updatedA), lambda)
  }

  private def matrixNormalization(B: DenseMatrix[Double]): DenseMatrix[Double] = {
    val A: DenseMatrix[Double] = B.copy
    val colNorms: DenseVector[Double] = norm(A(::, *)).t.toDenseVector

    for (i <- 0 until A.cols optimized) {
      A(::, i) :/= colNorms(i)
    }
    A
  }

  private def isConverged(oldA: DenseMatrix[Double],
                  newA: DenseMatrix[Double],
                  dotProductThreshold: Double): Boolean = {
    if (oldA == null || oldA.size == 0) {
      return false
    }

    val dprod = diag(oldA.t * newA)
    logger.info(s"dot(oldA, newA): ${diag(oldA.t * newA)}")

    all(dprod >:> dotProductThreshold)
  }
}
