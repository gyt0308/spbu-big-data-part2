package org.apache.spark.ml.lr

import org.scalatest.flatspec._
import org.scalatest.matchers._
import breeze.linalg.DenseVector
import org.apache.spark.ml.{Pipeline}
import org.apache.spark.ml.linalg.{ Vectors}
import org.apache.spark.sql.{ SparkSession, functions}

import java.io.File


class LinearRegressionTest extends AnyFlatSpec with should.Matchers{
  val spark = SparkSession.builder
    .appName("LinearRegressionTest")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._
  
  val delta = 0.0001
  val weightsInaccuracy = 0.1

  val firstValue = 13.5
  val oValue = 12

  val weights = Vectors.dense(1.1, -2.0, 1.7)
  val bias = 1.3

  val randPoints = Seq.fill(30)(Vectors.fromBreeze(DenseVector.rand(3)))
  val randData = randPoints.map(x => Tuple1(x)).toDF("features")

  "Model" should "work property" in {
    val vectors = Seq(
      Vectors.dense(firstValue, oValue, 7.0),
      Vectors.dense(-1, 0, 3.2),
    )
    val data = vectors.map(x => Tuple1(x)).toDF("features")
    var linearRegressionModel = new LinearRegressionModel(weights.toDense,bias)
    linearRegressionModel.setInputCol("features")
    linearRegressionModel.setOutputCol("prediction")

    val result = linearRegressionModel.transform(data).collect().map(_.getAs[Double](1))

    result(0) should be(vectors(0)(0) * weights(0) + vectors(0)(1) * weights(1) + vectors(0)(2) * weights(2) + bias +- delta)
    result(1) should be(vectors(1)(0) * weights(0) + vectors(1)(1) * weights(1) + vectors(1)(2) * weights(2) + bias +- delta)

  }


  "Estimator" should "work property" in {
    val linearRegressionModel = new LinearRegressionModel(weights.toDense, bias).setInputCol("features")
      .setOutputCol("label")
    import functions._
    val train = linearRegressionModel
      .transform(randData)
      .select(col("features"), (col("label") + rand() * lit(0.1) - lit(0.05)).as("label"))

    val linearRegression: LinearRegression = new LinearRegression(1, 1000)
    linearRegression.setInputCol("features")
    linearRegression.setOutputCol("label")
    val model = linearRegression.fit(train)
    val modelBias = model.bias
    val modelWeights = model.weights

    modelWeights(0) should be(weights(0) +- weightsInaccuracy)
    modelWeights(1) should be(weights(1) +- weightsInaccuracy)
    modelWeights(2) should be(weights(2) +- weightsInaccuracy)

    modelBias  should be(bias +- weightsInaccuracy)


    val true_model: LinearRegressionModel = new LinearRegressionModel(
      weights = weights.toDense,
      bias = bias
    ).setInputCol("features_test")
      .setOutputCol("label_test")

    val trainForTest = true_model
      .transform(randData.select(functions.col("features").as("features_test")))
      .select(col("features_test"), (col("label_test") + rand() * lit(0.1) - lit(0.05)).as("label_test"))

    val stepSize = 1.0
    val numIterations = 1000
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression(stepSize, numIterations)
        .setInputCol("features_test")
        .setOutputCol("label_test")
    ))

    pipeline.getStages(0).asInstanceOf[LinearRegression].stepSize should be(stepSize)
    pipeline.getStages(0).asInstanceOf[LinearRegression].numIterations should be(numIterations)

    var regressionModel = pipeline.fit(trainForTest).stages(0).asInstanceOf[LinearRegressionModel]
    var regressionModelWeights = regressionModel.weights
    var regressionModelBias = regressionModel.bias
    regressionModelWeights(0) should be(weights(0) +- weightsInaccuracy)
    regressionModelWeights(1) should be(weights(1) +- weightsInaccuracy)
    regressionModelWeights(2) should be(weights(2) +- weightsInaccuracy)
    regressionModelBias should be(bias +- weightsInaccuracy)

    //test re-read
    val dir = new File(System.getProperty("java.io.tmpdir") + File.separator + System.currentTimeMillis())
    pipeline.write.overwrite().save(dir.getAbsolutePath)
    regressionModel = Pipeline.load(dir.getAbsolutePath).fit(trainForTest).stages(0).asInstanceOf[LinearRegressionModel]
    regressionModelWeights = regressionModel.weights
    regressionModelBias = regressionModel.bias
    regressionModelWeights(0) should be(weights(0) +- weightsInaccuracy)
    regressionModelWeights(1) should be(weights(1) +- weightsInaccuracy)
    regressionModelWeights(2) should be(weights(2) +- weightsInaccuracy)
    regressionModelBias should be(bias +- weightsInaccuracy)
  }

}
