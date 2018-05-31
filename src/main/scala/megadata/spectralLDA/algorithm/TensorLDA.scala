package megadata.spectralLDA.algorithm

/**
 * Tensor Decomposition Algorithms.
 * Alternating Least Square algorithm is implemented.
 * Created by Furong Huang on 11/2/15.
 */
import breeze.linalg.{*, DenseMatrix, DenseVector, SparseVector, argsort, diag, norm}
import breeze.numerics._
import breeze.stats.distributions.{Rand, RandBasis}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector => mlVector}
import megadata.spectralLDA.datamoments.DataCumulant
import megadata.spectralLDA.utils.Datasets
import org.apache.log4j.Logger


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
                slackDimK: Option[Int] = None,
                postProcessing: Boolean = false) extends Serializable {
  @transient private lazy val logger = Logger.getLogger("TensorLDA")

  assert(dimK > 0, "The number of topics dimK must be positive.")
  assert(!slackDimK.isDefined || slackDimK.get >= 0,
    "slackDimK must be at least 0")

  // The following paper discusses about the success of random
  // projection with relation to the dimension of the sub-space
  // we're concerned about. i.e. To get the first dimK eigenvectors
  // we better project into (dimK + certain slack) sub-space.
  //
  // We give even more slack as the SVD of rescaled M2 is just one step
  // before the CP decomposition of rescaled M3. As one final discovered
  // topic could be the combination of multiple eigenvectors of M2, it's
  // better to allow sufficient redundancy in the number of eigenvectors
  // we compute for M2.

  // Ref:
  // Universality laws for randomized dimension reduction
  // with applications, S. Oymak and J. A. Tropp. Inform. Inference, Nov. 2017
  // Theorem II on Restricted Minimum Singular Value
  val slackK = slackDimK.getOrElse(dimK)
  logger.info(s"Slack of random projection dimension: $slackK")

  assert(alpha0 > 0, "The topic concentration alpha0 must be positive.")
  assert(maxIterations > 0, "The number of iterations for ALS must be positive.")
  assert(tol > 0.0, "tol must be positive and probably close to 0.")

  /** Run the fitting
    *
    * @param documents  RDD of the documents
    * @param randBasis  Random seed
    * @return           matrix of fitted topic-term distribution
    *                   (each column per topic),
    *                   vector of fitted topic prior distribution parameter,
    *                   eigenvectors of scaled M2 (column-wise),
    *                   eigenvalues of scaled M2,
    *                   vector of average term frequency
    */
  def fit(documents: RDD[(Long, SparseVector[Double])])
         (implicit randBasis: RandBasis = Rand)
          : (DenseMatrix[Double], DenseVector[Double],
             DenseMatrix[Double], DenseVector[Double],
             DenseVector[Double]) = {
    val cumulant: DataCumulant = DataCumulant.getDataCumulant(
      dimK + slackK,
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

  /** Runs the fitting for document vectors in Spark Mllib Vector format */
  def fit2(documents: RDD[(Long, mlVector)])
          (implicit randBasis: RandBasis = Rand)
  : (DenseMatrix[Double], DenseVector[Double],
    DenseMatrix[Double], DenseVector[Double],
    DenseVector[Double]) = {
    val mlDocuments = documents map {
      case (docid, v) => (docid, Datasets.mllibToBreeze(v))
    }
    fit(mlDocuments)(randBasis)
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

object TensorLDA {

  /** Describes topics with term indices
    *
    * @param beta              Matrix of fitted topic-term distribution
    *                          (each column per topic)
    * @param maxTermsPerTopic  Max terms per topic
    * @return                  Array of (term indices, weights) tuple
    */
  def describeTopics(beta: DenseMatrix[Double],
                     maxTermsPerTopic: Int)
  : Array[(Array[Int], Array[Double])] = {
    val topics =
      for (j <- 0 until beta.cols) yield {
        val topTermIds = argsort(beta(::, j)).reverse.take(maxTermsPerTopic)
        (topTermIds.toArray, beta(topTermIds, j).toArray)
      }
    topics.toArray
  }

  /** Describe and print topics with term words
    *
    * @param beta              Matrix of fitted topic-term distribution
    *                          (each column per topic)
    * @param idToWordMap       Map of word-id to word
    * @param maxTermsPerTopic  Max terms per topic
    * @return                  Array of (term words, weights) tuple
    */
  def describeTopicsInWords(beta: DenseMatrix[Double],
                            idToWordMap: Map[Int, String],
                            maxTermsPerTopic: Int)
  : Array[(Array[String], Array[Double])] = {
    val topics = describeTopics(beta, maxTermsPerTopic)
      .map {
        case (ids, weights) => (ids map idToWordMap, weights)
      }
    topics.zipWithIndex.foreach {
      case ((terms, _), i) =>
        println(s"Topic #$i: ${terms.mkString(", ")}")
    }
    topics
  }

}
