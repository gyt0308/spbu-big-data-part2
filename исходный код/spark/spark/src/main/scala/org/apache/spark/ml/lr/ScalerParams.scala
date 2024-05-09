package org.apache.spark.ml.lr

import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}


trait ScalerParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String) : Unit = set(inputCol, value)
  def setOutputCol(value: String): Unit = set(outputCol, value)
}
