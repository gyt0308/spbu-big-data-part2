package org.apache.spark.ml.lr

import breeze.linalg.sum
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
 * Расчет потерь и градиента
 * Используйте градиент наименьших квадратов
 */
class LSGradient  extends Serializable {
  def computeGradient(vector: Vector, label: Double, weightsVector: Vector): (Vector, Double) = {
    val product = vector.asBreeze *:* weightsVector.asBreeze
    val diff = sum(product) - label
    val gradient = vector.asBreeze * diff
    (Vectors.fromBreeze(gradient), (diff * diff) / 2.0)
  }
}
