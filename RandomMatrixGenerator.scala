import breeze.linalg.DenseMatrix
import breeze.stats.distributions.Gaussian

object RandomMatrixGenerator {
  def generate(rows: Int, cols: Int): DenseMatrix[Double] = {
    DenseMatrix.rand(rows, cols, Gaussian(0, 1))
  }
}
