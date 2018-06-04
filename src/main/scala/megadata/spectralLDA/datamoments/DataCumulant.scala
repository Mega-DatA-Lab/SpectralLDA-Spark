package megadata.spectralLDA.datamoments

/**
 * Data Cumulants Calculation.
 * Created by Furong Huang on 11/2/15.
 */

import megadata.spectralLDA.utils.Tensors
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.{Rand, RandBasis}
import megadata.spectralLDA.algorithm.RandNLA
import megadata.spectralLDA.utils.Tensors
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.log4j.Logger


/** Data cumulant
  *
  * @param thirdOrderMoments Scaled whitened M3, precisely,
  *                            <math>\frac{\alpha_0(\alpha_0+1)(\alpha_0+2)}{2} M3(W^T,W^T,W^T)</math>
  * @param eigenVectorsM2   V-by-k top eigenvectors of shifted M2, stored column-wise
  * @param eigenValuesM2    length-k top eigenvalues of shifted M2
  * @param firstOrderMoments  average of term count frequencies M1
  *
  * REFERENCES
  * [Wang2015] Wang Y et al, Fast and Guaranteed Tensor Decomposition via Sketching, 2015,
  *            http://arxiv.org/abs/1506.04448
  *
  */
case class DataCumulant(thirdOrderMoments: DenseMatrix[Double],
                        eigenVectorsM2: DenseMatrix[Double],
                        eigenValuesM2: DenseVector[Double],
                        firstOrderMoments: DenseVector[Double],
                        whiteningMatrix: DenseMatrix[Double],
                        whitenedWordTriplets: DenseMatrix[Double],
                        whitenedWordPairs: DenseMatrix[Double])
  extends Serializable


object DataCumulant {
  @transient private lazy val logger = Logger.getLogger("DataCumulant")

  def getDataCumulant(dimK: Int,
                      alpha0: Double,
                      documents: RDD[(Long, SparseVector[Double])],
                      randomisedSVD: Boolean = true,
                      numIterationsKrylovMethod: Int)
                     (implicit randBasis: RandBasis = Rand)
        : DataCumulant = {
    assert(dimK > 0, "The number of topics dimK must be positive.")
    assert(alpha0 > 0, "The topic concentration alpha0 must be positive.")

    val sc: SparkContext = documents.sparkContext

    val validDocuments = documents
      .map {
        case (id, wc) => (id, sum(wc), wc)
      }
      .filter {
        case (_, len, _) => len >= 3
      }
    validDocuments.cache()

    val dimVocab = validDocuments.map(_._3.length).take(1)(0)
    val numDocs = validDocuments.count()

    logger.info("Start calculating first order moments...")
    val firstOrderMoments = validDocuments
      .map {
        case (docId, len, vec) => vec * (1.0 / len)
      }
      .reduce(_ + _)
      .toDenseVector
      .*:*(1.0 / numDocs)
    logger.info("Finished calculating first order moments.")

    logger.info("Start calculating second order moments...")
    val (eigenVectors: DenseMatrix[Double], eigenValues: DenseVector[Double]) =
      if (randomisedSVD) {
        RandNLA.whiten2(
          alpha0,
          dimVocab,
          dimK,
          numDocs,
          firstOrderMoments,
          validDocuments,
          nIter = numIterationsKrylovMethod
        )
      }
      else {
        val E_x1_x2: DenseMatrix[Double] = validDocuments
          .map { case (_, len, vec) =>
            (Tensors.spVectorTensorProd2d(vec) - diag(vec)) / (len * (len - 1))
          }
          .reduce(_ + _)
          .map(_ / numDocs.toDouble).toDenseMatrix
        val M2: DenseMatrix[Double] = E_x1_x2 - alpha0 / (alpha0 + 1) * (firstOrderMoments * firstOrderMoments.t)

        val eigSym.EigSym(sigma, u) = eigSym(alpha0 * (alpha0 + 1) * M2)
        val i = argsort(sigma)
        (u(::, i.slice(dimVocab - dimK, dimVocab)).copy, sigma(i.slice(dimVocab - dimK, dimVocab)).copy)
      }
    logger.info("Finished calculating second order moments and whitening matrix.")

    logger.info("Start whitening data with dimensionality reduction...")
    val W: DenseMatrix[Double] = eigenVectors * diag(eigenValues map { x => 1 / (sqrt(x) + 1e-9) })

    // whitened document vectors plus the normalisation constants
    val whitenedDocs = validDocuments.map {
      case (docId, len, vec) =>
        (docId, len, vec, W.t * vec,
          1.0 / len / (len - 1), 1.0 / len / (len - 1) / (len - 2))
    }
    val whitenedM1 = W.t * firstOrderMoments
    logger.info("Finished whitening data.")

    // We computing separately the first order, second order, 3rd order terms in Eq (25) (26)
    // in [Wang2015]. For the 2nd order, 3rd order terms, We'd achieve maximum performance with
    // reduceByKey() of w_i, 1\le i\le V, the rows of the whitening matrix W.
    logger.info("Start calculating third order moments...")

    // Whitened word triplets
    val whitenedWordTripletsPart1 = whitenedDocs
      .map {
        case (_, _, _, p, _, c3) =>
          Tensors.makeRankOneTensor3d(p, p, p * c3)
      }
      .reduce(_ + _)

    // As of Spark 2.3.0, an efficient way to do
//    val wordTripletsPart2Mat = whitenedDocs
//      .map {
//        case (_, _, v, p, _, c3) =>
//          v.asCscColumn * (p * c3).t
//      }
//      .reduce(_ + _)
//      .toDenseMatrix
    val wordTripletsPart2Mat = DenseMatrix.zeros[Double](dimVocab, dimK)
    whitenedDocs
      .flatMap {
        case (_, _, v, p, _, c3) =>
          val z = p * c3
          v.activeIterator.map {
            case (wid, cnt) => (wid, z * cnt)
          }
      }
      .reduceByKey(_ + _)
      .collect
      .foreach {
        case (wid, y) => wordTripletsPart2Mat(wid, ::) := y.t
      }

    val whitenedWordTripletsPart2 = wordTripletsPart2Mat(*, ::).iterator
      .zip(W(*, ::).iterator)
      .map {
        case (v, w) =>
          (Tensors.makeRankOneTensor3d(v, w, w)
           + Tensors.makeRankOneTensor3d(w, v, w)
           + Tensors.makeRankOneTensor3d(w, w, v))
      }
      .reduce(_ + _)

    val wordTripletsPart3Vec = whitenedDocs
      .map {
        case (_, _, v, _, _, c3) => v * c3
      }
      .reduce(_ + _)
      .toDenseVector
    val whitenedWordTripletsPart3 = wordTripletsPart3Vec.valuesIterator
      .zip(W(*, ::).iterator)
      .map {
        case (v, w) =>
          v * Tensors.makeRankOneTensor3d(w, w, w)
      }
      .reduce(_ + _)

    val whitenedWordTriplets: DenseMatrix[Double] = (whitenedWordTripletsPart1
      - whitenedWordTripletsPart2 + 2.0 * whitenedWordTripletsPart3)
    whitenedWordTriplets :*= 1.0 / numDocs

    // whitened word pairs
    val whitenedWordPairsPart1 = whitenedDocs
      .map {
        case (_, _, _, p, c2, _) => p * p.t * c2
      }
      .reduce(_ + _)

    val whitenedWordPairsPart2Vec = whitenedDocs
      .map {
        case (_, _, v, _, c2, _) => v * c2
      }
      .reduce(_ + _)
    val whitenedWordPairsPart2 = W.t * diag(whitenedWordPairsPart2Vec) * W

    val whitenedWordPairs: DenseMatrix[Double] = (whitenedWordPairsPart1
      - whitenedWordPairsPart2)
    whitenedWordPairs :*= 1.0 / numDocs

    // whitened word pairs outer dot whitened M1
    val whitenedWordPairsDotM1 = makeSymTensor(whitenedWordPairs, whitenedM1)

    // whitened M1 triple dot
    val whitenedDotM1 = Tensors.makeRankOneTensor3d(whitenedM1,
      whitenedM1, whitenedM1)

    val whitenedM3 = (whitenedWordTriplets
       - whitenedWordPairsDotM1 * (alpha0 / (alpha0 + 2))
       + whitenedDotM1 * (2 * alpha0 * alpha0 / (alpha0 + 1) / (alpha0 + 2)))

    logger.info("Finished calculating third order moments.")

    // rescale whitened M3 to get the sum of tensor products
    // of topic-word vectors
    // <math>\sum_i alpha_i beta_i\otimes beta_i\otimes beta_i</math>
    new DataCumulant(
      whitenedM3 * (alpha0 * (alpha0 + 1) * (alpha0 + 2) / 2.0),
      eigenVectors,
      eigenValues,
      firstOrderMoments,
      whiteningMatrix = W,
      whitenedWordTriplets = whitenedWordTriplets,
      whitenedWordPairs = whitenedWordPairs
    )
  }

  /** Make unfolded symmetric tensor of m12 outer dot z
    *
    * We permute z to occupy 1st, 2nd, 3rd modes in three sub-products
    * and sum them to return the symmetric tensor.
    *
    * @param m12  DenseMatrix[Double] for the 1st-2rd mode
    * @param z    DenseVector[Double] for the 3rd mode
    * @return     DenseMatrix[Double] for the unfolded symmetric
    *             tensor of m12 outer dot z
    */
  private def makeSymTensor(m12: DenseMatrix[Double],
                            z: DenseVector[Double]): DenseMatrix[Double] = {
    val t1 = m12(*, ::).map {
      x => (x * z.t).toDenseVector
    }
    val t2 = m12(*, ::).map {
      x => (z * x.t).toDenseVector
    }
    val t3 = z * m12.toDenseVector.t

    t1 + t2 + t3
  }
}
