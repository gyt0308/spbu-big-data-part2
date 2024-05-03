import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import breeze.linalg.DenseMatrix

object LinearRegressionModel {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Linear Regression with Breeze").getOrCreate()

    // Генерация случайных данных
    val randomMatrix = RandomMatrixGenerator.generate(100000, 3)
    val data = spark.createDataFrame(randomMatrix(*, ::).iterator.toSeq.map(row => 
      (row(0), row(1), row(2), 1.5 * row(0) + 0.3 * row(1) - 0.7 * row(2) + scala.util.Random.nextGaussian()))).toDF("feature1", "feature2", "feature3", "label")

    // векторизованные функции
    val assembler = new VectorAssembler().setInputCols(Array("feature1", "feature2", "feature3")).setOutputCol("features")
    val output = assembler.transform(data)

    // Настройка модели линейной регрессии
    val lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features")
    val model = lr.fit(output)

    // Выходные параметры модели
    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

    spark.stop()
  }
}
