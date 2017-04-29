package edu.uci.eecs.spectralLDA.algorithm

/**
* Tensor Decomposition Algorithms.
* Alternating Least Square algorithm is implemented.
* Created by Furong Huang on 11/2/15.
*/

import edu.uci.eecs.spectralLDA.utils.{AlgebraUtil, TensorOps}
import breeze.linalg.{*, DenseMatrix, DenseVector, diag, max, min, norm, svd}
import breeze.stats.distributions.{Gaussian, Rand, RandBasis}

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

  /** Run Alternating Least Squares (ALS)
    *
    * @param randBasis   default random seed
    * @return            three dimK-by-dimK matrices with all the $beta_i$ as columns,
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

      println("Start ALS iterations...")
      var iter: Int = 0
      while ((iter == 0) || (iter < maxIterations &&
        !AlgebraUtil.isConverged(A_prev, A, dotProductThreshold = 1 - tol))) {
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

        println(s"iter $iter\tlambda: max ${max(lambda)}, min ${min(lambda)}")

        iter += 1
      }
      println("Finished ALS iterations.")

      reconstructedLoss = TensorOps.dmatrixNorm(tensor3D - A * diag(lambda) * TensorOps.krprod(C, B).t)
      println(s"Reconstructed loss: $reconstructedLoss\tOptimal reconstructed loss: $optimalReconstructedLoss")

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
    val updatedA = unfoldedM3 * TensorOps.krprod(C, B) * TensorOps.to_invert(C, B)
    val lambda = norm(updatedA(::, *)).t.toDenseVector
    (AlgebraUtil.matrixNormalization(updatedA), lambda)
  }
}
