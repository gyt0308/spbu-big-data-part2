package org.apache.spark.ml.lr

import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}

/**
 * Характеристики линейного регрессивного класса
 */
trait LinearRegressionParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  setDefault(inputCol, "features")
  setDefault(outputCol, "label")
}
