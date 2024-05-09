package org.apache.spark.ml.lr

import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}

class StandardScalerTest extends AnyFlatSpec with should.Matchers{
  val spark = SparkSession.builder
    .appName("StandardScalerTest")
    .master("local[*]")
    .getOrCreate()
  import spark.implicits._

  val delta = 0.0001
  val firstValue = 15.5
  val oValue = 14

  lazy val vectors: Seq[Vector] = Seq(
    Vectors.dense(firstValue, oValue),
    Vectors.dense(-1, 0)
  )
  lazy val data: DataFrame = vectors.map(x => Tuple1(x)).toDF("features")

  "Model and Estimator" should "work property" in {
    val means = Vectors.dense(2.0, -0.5).toDense
    val stds = Vectors.dense(1.5, 0.5).toDense
    val scalerModel = new StandardScalerModel(means,stds)
    scalerModel.setInputCol("features")
    scalerModel.setOutputCol("features")

    val result = scalerModel.transform(data)

    var vec = result.collect().map(_.getAs[Vector](0))

    vec(0)(0) should be((firstValue - 2.0) / 1.5 +- delta)
    vec(1)(0) should be((-1 - 2.0) / 1.5 +- delta)

    vec(0)(1) should be((oValue + 0.5) / 0.5 +- delta)
    vec(1)(1) should be((0 + 0.5) / 0.5 +- delta)


    val standardScaler = new StandardScaler()
    standardScaler.setInputCol("features")
    standardScaler.setOutputCol("features")

    //calculate means
    val model = standardScaler.fit(data)
    val modelMeans = model.means
    modelMeans(0) should be(vectors.map(_(0)).sum / vectors.length +- delta)
    modelMeans(1) should be(vectors.map(_(1)).sum / vectors.length +- delta)

    //calculate stds
    val modelStds = model.stds
    modelStds(0) should be(Math.sqrt(vectors.map(v => (v(0) - modelMeans(0)) * (v(0) - modelMeans(0))).sum / (vectors.length - 1)) +- delta)
    modelStds(1) should be(Math.sqrt(vectors.map(v => (v(1) - modelMeans(1)) * (v(1) - modelMeans(1))).sum / (vectors.length - 1)) +- delta)

    //transform data
    vec = model.transform(data).collect().map(_.getAs[Vector](0))

    vec(0)(0) should be((firstValue - modelMeans(0)) / modelStds(0) +- delta)
    vec(1)(0) should be((-1 - modelMeans(0)) / modelStds(0) +- delta)

    vec(0)(1) should be((oValue - modelMeans(1)) / modelStds(1) +- delta)
    vec(1)(1) should be((0 - modelMeans(1)) / modelStds(1) +- delta)

    val standardScalerModel = new Pipeline().setStages(Array(
      standardScaler
    )).fit(data).stages(0).asInstanceOf[StandardScalerModel]

    val standardScalerModelMeans = standardScalerModel.means
    standardScalerModelMeans(0) should be(vectors.map(_(0)).sum / vectors.length +- delta)
    standardScalerModelMeans(1) should be(vectors.map(_(1)).sum / vectors.length +- delta)
  }



}
