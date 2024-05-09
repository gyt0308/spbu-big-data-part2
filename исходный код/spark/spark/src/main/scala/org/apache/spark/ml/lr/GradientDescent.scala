package org.apache.spark.ml.lr

import breeze.linalg.norm
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.{Dataset, Encoder, Row}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.expressions.Aggregator
import scala.collection.mutable.ArrayBuffer

/**
 * Полный градиентный спуск
 */
class GradientDescent (var gradient: LSGradient,
                                  var updater: SimpleUpdater,
                                  var inputCol: String,
                                  var outputCol: String)
  extends Logging with Serializable{

  private var stepSize: Double = 0.001
  private var numIterations: Int = 100

  def setStepSize(step: Double): Unit= {
    assert(step > 0,"step must be positive")
    this.stepSize = step
  }

  def setInputCol(inputCol: String): Unit = {
    this.inputCol = inputCol
  }

  def setOutputCol(outputCol: String): Unit = {
    this.outputCol = outputCol
  }

  def setNumIterations(iterations: Int): Unit = {
    assert(iterations >= 0, s"iterations must be positive")
    this.numIterations = iterations
  }

  def optimize(dataset: Dataset[_], initialWeights: Vector,convergenceTol:Double = 0.001): Vector = {
    //cache data
    dataset.persist()
    val LossBuffer = new ArrayBuffer[Double](numIterations)
    var bestLoss = Double.MaxValue
    var badItersCount = 0
    var preWeights: Option[Vector] = None
    var curWeights: Option[Vector] = None
    val rowsCount = dataset.count()

    var weights = Vectors.dense(initialWeights.toArray)
    var bestWeights = weights

    var iteration = 0
    //Индикатор, что решение работает
    var hasConverged = false
    while (!hasConverged && iteration <= numIterations) {
      iteration += 1
      //Рассмотрим общий градиент и потерю
      val summerAggregator = new Aggregator[Row, (Vector, Double), (Vector, Double)] {
        def zero: (Vector, Double) = (Vectors.zeros(weights.size), 0.0)

        def reduce(acc: (Vector, Double), x: Row): (Vector, Double) = {
          val (grad, loss) = gradient.computeGradient(x.getAs[Vector](inputCol), x.getAs[Double](outputCol), weights)
          val vector = Vectors.fromBreeze(acc._1.asBreeze + grad.asBreeze / rowsCount.asInstanceOf[Double])
          val value = acc._2 + loss / rowsCount.asInstanceOf[Double]

          (vector,value)
        }

        def merge(acc1: (Vector, Double), acc2: (Vector, Double)): (Vector, Double) = {
          val vector = Vectors.fromBreeze(acc1._1.asBreeze + acc2._1.asBreeze)
          val value = acc1._2 + acc2._2

          (vector,value)
        }

        def finish(r: (Vector, Double)): (Vector, Double) = r

        override def outputEncoder: Encoder[(Vector, Double)] = ExpressionEncoder()

        override def bufferEncoder: Encoder[(Vector, Double)] = outputEncoder
      }.toColumn
      val firstRow = dataset.select(summerAggregator.as[(Vector, Double)](ExpressionEncoder())).first()
      weights = updater.compute(weights, firstRow._1, stepSize)
      val loss = firstRow._2
      LossBuffer += loss
      preWeights = curWeights
      curWeights = Some(weights)
      if (loss >= bestLoss) {
        badItersCount += 1
      } else {
        bestWeights = weights
        bestLoss = loss
        badItersCount = 0
      }
      if (!preWeights.isEmpty && !curWeights.isEmpty) {
        if (convergenceTol != 0.0) {
          //Определение сходимости решения
          val previousBDV = preWeights.get.asBreeze.toDenseVector
          val currentBDV = curWeights.get.asBreeze.toDenseVector
          val solutionVecDiff = norm(previousBDV - currentBDV)
          hasConverged = solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
        } else {
          //badItersCount должен <= 5
          hasConverged = badItersCount > 5
        }
      }
    }

    dataset.unpersist()
    weights
  }
}
