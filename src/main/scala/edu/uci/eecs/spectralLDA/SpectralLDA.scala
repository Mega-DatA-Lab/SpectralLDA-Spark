 /**
 * Learning LDA model through inverse method of moments (moment matching).
 * Decomposition of third order moment (tensor) finds the model parameter.
 * Created by Furong Huang on 11/2/15.
 */

package edu.uci.eecs.spectralLDA

import edu.uci.eecs.spectralLDA.algorithm.TensorLDA
import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, sum}
import org.apache.spark.{SparkConf, SparkContext}

import scalaxy.loops._
import scala.language.postfixOps
import scopt.OptionParser
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import java.io._
import java.nio.file.{Files, Paths}


object SpectralLDA {
  private case class Params(
                             input: Seq[String] = Seq.empty,//use this for customized input
                             inputType: String = "obj", // "libsvm", "text" or "obj"
                             k: Int = 1,
                             topicConcentration: Double = 5.0,
                             maxIterations: Int = 500,
                             tolerance: Double = 1e-6,
                             outputDir: String = "."
                          )

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()

    val parser: OptionParser[Params] = new OptionParser[Params]("SpectralLDA") {
      head("Spectral LDA Factorization")

      opt[Int]('k', "k").required()
        .text("number of topics")
        .action((x, c) => c.copy(k = x))
        .validate(x =>
          if (x > 0) success
          else failure("The number of topics k must be positive.")
        )
      opt[Double]("alpha0").required()
        .text("sum of the topic distribution prior parameter")
        .action((x, c) => c.copy(topicConcentration = x))
        .validate(x =>
          if (x > 0.0) success
          else failure("topicConcentration must be positive.")
        )

      opt[Int]("max-iter")
        .text(s"number of iterations of ALS. default: ${defaultParams.maxIterations}")
        .action((x, c) => c.copy(maxIterations = x))
        .validate(x =>
          if (x > 0) success
          else failure("maxIterations must be positive.")
        )
      opt[Double]("tol")
        .text(s"tolerance for the ALS algorithm. default: ${defaultParams.tolerance}")
        .action((x, c) => c.copy(tolerance = x))
        .validate(x =>
          if (x > 0.0) success
          else failure("tolerance must be positive.")
        )

      opt[String]("input-type")
        .text(s"""type of input files: "obj", "libsvm" or "text". "obj" for serialised RDD[(Long, SparseVector[Double])] file. default: ${defaultParams.inputType}""")
        .action((x, c) => c.copy(inputType = x))
        .validate(x =>
          if (x == "obj" || x == "libsvm" || x == "text") success
          else failure("""inputType must be "obj", "libsvm" or "text".""")
        )
      opt[String]('o', "output-dir").valueName("<dir>")
        .text(s"output write path. default: ${defaultParams.outputDir}")
        .action((x, c) => c.copy(outputDir = x))
        .validate(x =>
          if (Files.exists(Paths.get(x))) success
          else failure(s"Output directory $x doesn't exist.")
        )

      help("help").text("prints this usage text")

      arg[String]("<input>...")
        .text("paths of input files")
        .unbounded()
        .required()
        .action((x, c) => c.copy(input = c.input :+ x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) =>
        run(params)
      case None =>
        parser.showUsageAsError
        sys.exit(1)
    }
  }


  private def run(params: Params): (RDD[(Long, SparseVector[Double])], Array[String], DenseMatrix[Double], DenseVector[Double]) = {

    Logger.getRootLogger.setLevel(Level.WARN)
    if (params.inputType == "libsvm") {
      println("Input data in libsvm format.")
    }
    else {
      println("Converting raw text to libsvm format.")
    }

    val applicationStart: Long = System.nanoTime()
    val preprocessStart: Long = System.nanoTime()

    val conf: SparkConf = new SparkConf().setAppName(s"Spectral LDA via Tensor Decomposition: $params")
    val sc: SparkContext = new SparkContext(conf)
    println("Generated the SparkConetxt")

    println("Start reading data...")
    val (documents: RDD[(Long, SparseVector[Double])], vocabArray: Array[String]) = params.inputType match {
      case "obj" =>
        (sc.objectFile[(Long, SparseVector[Double])](params.input.mkString(",")), Array[String]())
    }
    println("Finished reading data.")

    println("Start ALS algorithm for tensor decomposition...")
    val lda = new TensorLDA(
      params.k,
      params.topicConcentration,
      maxIterations = params.maxIterations,
      tol = params.tolerance
    )
    val (beta, alpha, _, _, _) = lda.fit(documents)
    println("Finished ALS algorithm for tensor decomposition.")

    val preprocessElapsed: Double = (System.nanoTime() - preprocessStart) / 1e9
    val numDocs: Double = documents.countApprox(30000L).getFinalValue.mean
    val dimVocab: Int = documents.map(_._2.length).take(1)(0)
    sc.stop()
    println()
    println("Corpus summary:")
    println(s"\t Training set size: ~$numDocs documents")
    println(s"\t Vocabulary size: $dimVocab terms")
    println(s"\t Model Training time: $preprocessElapsed sec")
    println()

    //time
    val thisK = params.k
    val applicationElapsed: Double = (System.nanoTime() - applicationStart) / 1e9
    val writer_time = new PrintWriter(new File(params.outputDir + s"/TD_runningTime_k$thisK" + ".txt"))
    writer_time.write(s"$applicationElapsed sec")
    writer_time.close()

    println()
    println("Learning done. Writing topic word matrix (beta) and topic proportions (alpha)... ")

    // beta
    breeze.linalg.csvwrite(new File(params.outputDir + s"/TD_beta_k$thisK" + ".txt"), beta, separator = ' ')

    //alpha
    // println(alpha.map(x => math.abs(x/alpha0Estimate*params.topicConcentration)))
    val alpha0Estimate:Double = breeze.linalg.sum(alpha)
    val writer_alpha = new PrintWriter(new File(params.outputDir + s"/alpha_k$thisK" + ".txt" ))
    for( i <- 0 until alpha.length optimized){
      var thisAlpha: Double = alpha(i) / alpha0Estimate * params.topicConcentration
      writer_alpha.write(s"$thisAlpha \t")
    }
    writer_alpha.close()

    (documents, vocabArray, beta, alpha)
  }
}
