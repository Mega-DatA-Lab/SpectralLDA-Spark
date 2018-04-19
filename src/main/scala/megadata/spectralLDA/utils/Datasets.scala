package megadata.spectralLDA.utils

import breeze.linalg.{SparseVector => brSparseVector}
import org.apache.spark.mllib.linalg.{Vector => mlVector, Vectors => mlVectors}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/** Utility functions to read various datasets */
object Datasets {

  /** Read UCI Bag of Words Dataset
    *
    * The output still remains in tuples (doc-id, (word-id, count)).
    * We have to call [[uciBowFeaturesToBreeze]] or [[uciBowFeaturesToMllib]]
    * to produce the documents for the topic learning class. In this way
    * the user can cache the output of this function and run the
    * Tensor LDA, Spark LDA in the same session.
    *
    * @param sc               SparkContext
    * @param docWordFilePath  docword file path
    * @param vocabFilePath    vocab file path
    * @param maxFeatures      Max number of features to retain
    *                         for the bag of words
    * @return                 RDD of (doc-id, (word-id, count)),
    *                         Array of (vocabulary word, index)
    */
  def readUciBagOfWords(sc: SparkContext,
                        docWordFilePath: String,
                        vocabFilePath: String,
                        maxFeatures: Int)
  : (RDD[(Long, (Int, Double))], Array[(String, Int)]) = {
    val docWord = sc.textFile(docWordFilePath)

    // Stats of the doc-word dataset
    val docWordStats = docWord.take(3).map(_.toInt)
      match {
        case Array(docCount, vocabSize, nnz) => (docCount, vocabSize, nnz)
      }

    // Cast the doc-word tuples into integers
    // Change the 1-based word indexing to 0-based
    val docWordElements = docWord
      .map(_.split(" "))
      .filter(_.length == 3)
      .map {
        case Array(docid: String, wid: String, c: String)
        => (docid.toLong, wid.toInt - 1, c.toDouble)
      }

    // Document frequency of words
    val df = docWordElements
      .map {
        case (docid: Long, wid: Int, c: Double)
          => (wid, 1)
      }
      .reduceByKey(_ + _)

    // Sort DF and take the top maxFeatures words
    val dfSorted: RDD[(Int, Int)] = df.sortBy(- _._2)
    val wordIdsInFeatures: Array[Int] = dfSorted
      .map(_._1).take(maxFeatures)

    // Filter the doc-word dataset with the words in features
    val docWordElementsInFeatures = docWordElements
      .filter(wordIdsInFeatures contains _._2)
      .cache()

    // Reindex words in features
    val vocab = sc.textFile(vocabFilePath)
      .zipWithIndex()
      .map {
        case (w, wid) => (w, wid.toInt)
      }
    val vocabFeatures = vocab
      .filter(wordIdsInFeatures contains _._2)
      .zipWithIndex()
      .map {
        case ((w, wid), newid) => (w, wid, newid.toInt)
      }

    // Reindex the words in features
    val features = docWordElementsInFeatures
      .map {
        case (docid, wid, c) => (wid, (docid, c))
      }
      .join(vocabFeatures.map {
        case (w, wid, newid) => (wid, newid)
      })
      .map {
        case (wid, ((docid, c), newid)) => (docid, (newid, c))
      }

    (features, vocabFeatures.map(x => (x._1, x._3)).collect)
  }

  /** Convert bag-of-word tuples to documents for TensorLDA
    *
    * @param features     RDD of (doc-id, (word-id, count))
    * @param maxFeatures  Max number of features
    * @return             RDD of (doc-id, Breeze Sparse Vector)
    */
  def uciBowFeaturesToBreeze(features: RDD[(Long, (Int, Double))],
                             maxFeatures: Int)
  : RDD[(Long, brSparseVector[Double])] = {
    features
      .groupByKey()
      .mapValues {
        x => brSparseVector[Double](maxFeatures)(x.toSeq: _*)
      }
  }

  /** Convert bag-of-word tuples to documents for Spark LDA
    *
    * @param features     RDD of (doc-id, (word-id, count))
    * @param maxFeatures  Max number of features
    * @return             RDD of (doc-id, Spark Mllib Vector)
    */
  def uciBowFeaturesToMllib(features: RDD[(Long, (Int, Double))],
                            maxFeatures: Int)
  : RDD[(Long, mlVector)] = {
    features
      .groupByKey()
      .mapValues {
        x => mlVectors.sparse(maxFeatures, x.toSeq)
      }
  }

  /** Convert a Breeze Sparse Vector to Spark Mllib Vector */
  def breezeToMllib(v: brSparseVector[Double]): mlVector = {
    mlVectors.sparse(v.length, v.activeIterator.toSeq)
  }

  /** Convert a Spark Mllib Vector to Breeze Sparse Vector */
  def mllibToBreeze(v: mlVector): brSparseVector[Double] = {
    val vSparse = v.toSparse
    new brSparseVector[Double](vSparse.indices, vSparse.values, v.size)
  }

}