import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

object LinearRegressionApp {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Linear Regression with Breeze and Spark ML")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // 生成随机数据
    val data = RandomMatrixGenerator.generate(100000, 3)
    val df = spark.createDataFrame(data.map(row => (row(0), row(1), row(2), 1.5 * row(0) + 0.3 * row(1) - 0.7 * row(2) + Gaussian(0, 1).draw()))).toDF("feature1", "feature2", "feature3", "label")

    val assembler = new VectorAssembler().setInputCols(Array("feature1", "feature2", "feature3")).setOutputCol("features")
    val finalData = assembler.transform(df).select("features", "label")

    val lr = new LinearRegression().setMaxIter(10).setRegParam(0.01)
    val model = lr.fit(finalData)

    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

    spark.stop()
  }
}
