package org.apache.spark.ml.lr

import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.util._
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType


class StandardScaler(override val uid: String) extends Estimator[StandardScalerModel] with ScalerParams
with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("standardScaler"))

  override def fit(dataset: Dataset[_]): StandardScalerModel = {
    val firstRow = dataset
      .select(Summarizer.metrics("mean", "std").summary(dataset($(inputCol))))
      .first()
    val mean = firstRow.getAs[GenericRowWithSchema]("aggregate_metrics(features, 1.0)").apply(0)
    val std = firstRow.getAs[GenericRowWithSchema]("aggregate_metrics(features, 1.0)").apply(1)
    copyValues(new StandardScalerModel(mean.asInstanceOf[Vector].toDense, std.asInstanceOf[Vector].toDense)).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[StandardScalerModel] = null

  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains($(outputCol))) {
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}
object StandardScaler extends DefaultParamsReadable[StandardScaler]


class StandardScalerModel private[lr](
                           override val uid: String,
                           val means: DenseVector,
                           val stds: DenseVector) extends Model[StandardScalerModel] with ScalerParams {

  private[lr] def this(means: DenseVector, stds: DenseVector) =
    this(Identifiable.randomUID("standardScalerModel"), means, stds)

  override def copy(extra: ParamMap): StandardScalerModel = null

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transform = dataset.sqlContext.udf.register(uid + "_transform",
      (x : Vector) => {
         Vectors.fromBreeze((x.asBreeze - means.asBreeze) /:/ stds.asBreeze)
      })

    dataset.withColumn($(outputCol), transform(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains($(outputCol))) {
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}
