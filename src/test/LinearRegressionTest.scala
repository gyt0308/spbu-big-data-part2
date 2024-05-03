import org.scalatest.FunSuite

class LinearRegressionTest extends FunSuite {
  test("Random matrix generation") {
    val matrix = RandomMatrixGenerator.generate(100, 3)
    assert(matrix.rows == 100 && matrix.cols == 3)
  }

}
