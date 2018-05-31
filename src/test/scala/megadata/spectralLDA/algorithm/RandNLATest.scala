package megadata.spectralLDA.algorithm

import breeze.linalg._
import breeze.linalg.qr.QR
import breeze.stats.distributions.{Gaussian, RandBasis, ThreadLocalRandomGenerator, Uniform}
import megadata.spectralLDA.testharness.Context
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.SparkContext
import org.scalatest._


class RandNLATest extends FlatSpec with Matchers {

  private val sc: SparkContext = Context.getSparkContext

  "Randomised Power Iteration method" should "be approximately correct" in {
    implicit val randBasis: RandBasis =
      new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(234787)))

    val n = 100
    val k = 5

    val alpha: DenseVector[Double] = DenseVector[Double](25.0, 20.0, 15.0, 10.0, 5.0)
    val beta: DenseMatrix[Double] = DenseMatrix.rand(n, k, Uniform(0.0, 1.0))

    val norms = norm(beta(::, *)).t.toDenseVector
    for (j <- 0 until k) {
      beta(::, j) /= norms(j)
    }

    val a: DenseMatrix[Double] = beta * diag(alpha) * beta.t
    val sigma: DenseMatrix[Double] = DenseMatrix.rand(n, k, Gaussian(mu = 0.0, sigma = 1.0))
    val y = a * sigma
    val QR(q: DenseMatrix[Double], _) = qr.reduced(y)

    val (s: DenseVector[Double], u: DenseMatrix[Double]) = RandNLA.decomp2(a * q, q)

    val diff_a = u * diag(s) * u.t - a
    val norm_diff_a = norm(norm(diff_a(::, *)).t.toDenseVector)

    norm_diff_a should be <= 1e-8
  }
}

