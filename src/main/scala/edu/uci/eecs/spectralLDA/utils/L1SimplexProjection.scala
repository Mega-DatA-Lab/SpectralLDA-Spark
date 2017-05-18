package edu.uci.eecs.spectralLDA.utils

import breeze.linalg.{DenseMatrix, DenseVector, max, min}

import scala.util.control.Breaks._
import scalaxy.loops._
import scala.language.postfixOps

object L1SimplexProjection {
  /** Projection of a vector onto a simplex
    *
    * Given a length-n vector V, find a vector W=(w_i)_{1\le i\le n} in the simplex that
    * \sum_{i=1}^n w_i=1, w_i>0 \forall i, by minimising the Euclidean distance between V and W.
    *
    * Ref:
    * Duchi, John, Efficient Projections onto the l1-Ball for Learning in High Dimensions, 2008
    *
    * @param V  The input vector
    * @return   Projected vector
    */
  def project(V: DenseVector[Double]): DenseVector[Double] = {
    // val z:Double = 1.0
    val len: Int = V.length
    val U: DenseVector[Double] = DenseVector(V.copy.toArray.sortWith(_ > _))
    val cums: DenseVector[Double] = DenseVector(AlgebraUtil.Cumsum(U.toArray).map(x => x-1))
    val Index: DenseVector[Double] = DenseVector((1 to (len + 1)).toArray.map(x => 1.0/x.toDouble))
    val InterVec: DenseVector[Double] = cums :* Index
    val TobefindMax: DenseVector[Double] = U - InterVec
    var maxIndex : Int = 0
    // find maxIndex
    breakable{
      for (i <- 0 until len optimized){
        if (TobefindMax(len - i - 1) > 0){
          maxIndex = len - i - 1
          break()
        }
      }
    }
    val theta: Double = InterVec(maxIndex)
    val P_norm: DenseVector[Double] = max(V - theta, 0.0)
    P_norm
  }
}