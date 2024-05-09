package org.apache.spark.ml.lr

import breeze.linalg.{Vector => BV, axpy => brzAxpy}
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
 * Обновление весов без изменения скорости обучения
 */
class SimpleUpdater  extends Serializable {
   def compute(weightsOld: Vector,
                       gradient: Vector,
                       stepSize: Double,
                       ): Vector = {

    val weights  = weightsOld.asBreeze.toDenseVector.asInstanceOf[BV[Double]]
    brzAxpy(-stepSize, gradient.asBreeze, weights)
    Vectors.fromBreeze(weights)
  }
}