package edu.uci.eecs.spectralLDA.utils

import edu.uci.eecs.spectralLDA.testharness.Context
import org.scalatest._
import java.nio.file.Files

class DatasetsTest extends FlatSpec with Matchers {

  private val sc = Context.getSparkContext

  "Re-indexed vocabulary" should "be correct" in {
    val docWordFilePath = Files.createTempFile("docword", "txt")
    val vocabFilePath = Files.createTempFile("vocab", "txt")

    val docWord = ("3\n3\n9\n"
      + "1 1 13\n1 2 1\n1 3 1\n"
      + "2 2 2\n2 3 1\n3 3 2\n")
    val vocab = "a\nb\nc\n"
    Files.write(docWordFilePath,
      java.util.Arrays.asList(docWord.split("\n"): _*))
    Files.write(vocabFilePath, vocab.getBytes)

    val (docs, newVocab) = Datasets.readUciBagOfWords(sc,
      docWordFilePath.toString, vocabFilePath.toString, maxFeatures = 2)

    val docsWithVocab = docs
      .map {
        case (docid, (wid, c)) => (wid, (docid, c))
      }
      .join(sc.parallelize(newVocab)
          .map {
            case (w, wid) => (wid, w)
          })
      .map {
        case (wid, ((docid, c), w)) => (docid, w, c)
      }

    docsWithVocab.collect.toSet should be (Set(
      (1, "c", 1.0), (2, "c", 1.0), (3, "c", 2.0),
      (1, "b", 1.0), (2, "b", 2.0)
    ))
  }

}