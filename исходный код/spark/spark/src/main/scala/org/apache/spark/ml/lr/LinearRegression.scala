package org.apache.spark.ml.lr

import breeze.linalg.sum
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.{Estimator, Model, lr}
import org.apache.spark.sql.{DataFrame, Dataset, Row, functions}
import org.apache.spark.sql.types.StructType

// Класс реализующий функционал линейной регрессии
class LinearRegression(override val uid: String,
                       val stepSize: Double,
                       val numIterations: Int) extends Estimator[LinearRegressionModel]
    with LinearRegressionParams
    with DefaultParamsWritable
    with MLWritable {

  def this(stepSize: Double, numIterations: Int) = this(
    Identifiable.randomUID("linearRegression"),
    stepSize,
    numIterations)

  private val gradient = new LSGradient()
  private val updater = new SimpleUpdater()
  private val optimizer = new GradientDescent(gradient, updater, $(inputCol), $(outputCol))

  optimizer.setStepSize(stepSize)
  optimizer.setNumIterations(numIterations)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    var weights: Vector = Vectors.dense(1.0, 1.0, 1.0, 1.0)  // инициализируем веса модели
    // Добавьте коэффициент свободы в противовес
    val litOne = dataset.withColumn("litOne", functions.lit(1))
    val assembled = new VectorAssembler()
      .setInputCols(Array($(inputCol), "litOne"))
      .setOutputCol("featuresOutCol")
      .transform(litOne)
      .select(functions.col("featuresOutCol").as($(inputCol)), functions.col($(outputCol)))
    //оптимизация
    weights = optimizer.optimize(assembled, weights)
    //Результаты анализа модели
    val modelWeights = new DenseVector(
      weights.toArray.slice(0, weights.size - 1))
    val bias =  weights.toArray(weights.size - 1)
    copyValues(new LinearRegressionModel(modelWeights,bias)).setParent(this)
  }

  override def copy(paramMap: ParamMap)= {
    copyValues(new LinearRegression(stepSize, numIterations))
  }

  // Передача новых названий столбцов оптимайзеру
  override def setInputCol(value: String): this.type = {
    set(inputCol, value)
    optimizer.setInputCol($(inputCol))
    this
  }

  override def setOutputCol(value: String): this.type = {
    set(outputCol, value)
    optimizer.setOutputCol($(outputCol))
    this
  }

  override def transformSchema(structType: StructType) = {
    if (structType.fieldNames.contains($(outputCol))) {
      structType
    } else {
      SchemaUtils.appendColumn(structType, structType(getInputCol).copy(name = getOutputCol))
    }
  }

  override def write: MLWriter = new LinearRegression.Writer(this)
}

/**
 * Статические члены класса LinearRegression
 */
object LinearRegression extends DefaultParamsReadable[LinearRegression] with MLReadable[LinearRegression] {
  override def read: MLReader[LinearRegression] = new LinearRegressionReader

  override def load(str: String): LinearRegression = super.load(str)

  private class Writer(linearRegression: LinearRegression) extends MLWriter {

    private case class Data(stepSize: Double, numIterations: Int)

    override protected def saveImpl(path: String): Unit = {
      // Сохранить метаданные и параметры
      DefaultParamsWriter.saveMetadata(linearRegression, path, sc)
      sparkSession
        .createDataFrame(Seq(Data(linearRegression.stepSize, linearRegression.numIterations)))
        .write
        .parquet(new Path(path, "data").toString)
    }
  }

  private class LinearRegressionReader extends MLReader[LinearRegression] {
    override def load(path: String): LinearRegression = {
      // Загружаем другие данные
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath).select("stepSize", "numIterations").head()
      val stepSize = data.getAs[Double](0)
      val i = data.getAs[Int](1)
      // Загрузка метаданных и параметров
      val metadata = DefaultParamsReader.loadMetadata(path, sc, classOf[LinearRegression].getName)
      // Создаём эстиматор с загруженными параметрами
      val transformer = new LinearRegression(metadata.uid, stepSize, i)
      metadata.getAndSetParams(transformer)
      transformer.optimizer.setInputCol(transformer.getInputCol)
      transformer.optimizer.setOutputCol(transformer.getOutputCol)
      transformer
    }
  }
}

/**
  Линейная регрессивная модель
**/
class LinearRegressionModel (
                           override val uid: String,
                           val weights: DenseVector,
                           val bias: Double)
  extends Model[LinearRegressionModel]
    with LinearRegressionParams
    with MLWritable {

   def this(weights: DenseVector, bias: Double) = this(Identifiable.randomUID("linearRegressionModel"), weights, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = {
    copyValues(new LinearRegressionModel(weights, bias))
  }

  /**
   * Получаем прогнозы с помощью модели
   * @param dataset Датасет
   * @return Исходный датасет с прогнозами
   */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformFunc = (x: Vector) => {
      sum(x.asBreeze *:* weights.asBreeze) + bias
    }

    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",transformFunc)

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(structType: StructType): StructType = {
    if (structType.fieldNames.contains($(outputCol))) {
      structType
    } else {
      SchemaUtils.appendColumn(structType, structType(getInputCol).copy(name = getOutputCol))
    }
  }

  override def write: MLWriter = new LinearRegressionModel.LinearRegressionModelWriter(this)
}


object LinearRegressionModel extends DefaultParamsReadable[LinearRegressionModel]
  with MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new LinearRegressionModelReader

  override def load(path: String): LinearRegressionModel = super.load(path)

  class LinearRegressionModelWriter(instance: LinearRegressionModel) extends MLWriter {

    private case class Data(weights: DenseVector, bias: Double)

    override protected def saveImpl(path: String): Unit = {
      // Сохранить метаданные и параметры
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      // Сохраняем другие данные
      val dataPath = new Path(path, "data").toString
      sparkSession
        .createDataFrame(Seq(Data(instance.weights, instance.bias)))
        .write
        .parquet(dataPath)
    }
  }

  class LinearRegressionModelReader extends MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      // Загружаем метаданные и параметры
      val metadata = DefaultParamsReader.loadMetadata(path, sc, classOf[LinearRegressionModel].getName)
      val dataPath = new Path(path, "data").toString
      val firstRow = sparkSession.read.parquet(dataPath).select("weights", "bias").head()
      // Создание модели
      val model = new LinearRegressionModel(metadata.uid, firstRow.getAs[DenseVector](0), firstRow.getAs[Double](1))
      metadata.getAndSetParams(model)
      model
    }
  }
}



