package edu.uci.eecs.spectralLDA.utils

import breeze.linalg.{SparseVector => brSparseVector}
import org.apache.spark.mllib.linalg.{Vector => mlVector, Vectors => mlVectors}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

object Datasets {

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

  def uciBowFeaturesToBreeze(features: RDD[(Long, (Int, Double))],
                             maxFeatures: Int)
  : RDD[(Long, brSparseVector[Double])] = {
    features
      .groupByKey()
      .mapValues {
        x => brSparseVector[Double](maxFeatures)(x.toSeq: _*)
      }
  }

  def uciBowFeaturesToMllib(features: RDD[(Long, (Int, Double))],
                            maxFeatures: Int)
  : RDD[(Long, mlVector)] = {
    features
      .groupByKey()
      .mapValues {
        x => mlVectors.sparse(maxFeatures, x.toSeq)
      }
  }

  def breezeToMllib(v: brSparseVector[Double]): mlVector = {
    mlVectors.sparse(v.length, v.activeIterator.toSeq)
  }

  def mllibToBreeze(v: mlVector): brSparseVector[Double] = {
    val vSparse = v.toSparse
    new brSparseVector[Double](vSparse.indices, vSparse.values, v.size)
  }

}