import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.sql.SparkSession

class LinearRegressionTest extends AnyFunSuite {
  test("RandomMatrixGenerator produces correct size") {
    val matrix = RandomMatrixGenerator.generate(100, 3)
    assert(matrix.rows == 100 && matrix.cols == 3)
  }

  test("Linear regression model training") {
    val spark = SparkSession.builder
      .appName("Linear Regression Test")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    val data = Seq(
      (1.0, 2.0, 3.0, 1.5),
      (4.0, 5.0, 6.0, 2.5),
      (7.0, 8.0, 9.0, 3.5)
    ).toDF("feature1", "feature2", "feature3", "label")

    val assembler = new VectorAssembler().setInputCols(Array("feature1", "feature2", "feature3")).setOutputCol("features")
    val finalData = assembler.transform(data).select("features", "label")

    val lr = new LinearRegression().setMaxIter(10).setRegParam(0.01)
    val model = lr.fit(finalData)

    assert(model.coefficients.size == 3)
    spark.stop()
  }
}