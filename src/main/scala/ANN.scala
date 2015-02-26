import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{ DenseMatrix, DenseVector, Vector, Vectors }
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.optimization._

class ANNmodel {

}
private class ANNLeastSquaresGradient{
  
}
private class ANNUpdater extends Updater {
  
}
class ANN(
  topology: Array[Int],
  maxNumIterations: Int,
  convergenceTol: Double,
  batchSize: Int = 1)extends Serializable  {

  
  
}
object ANN {

  private val defaultTolerance: Double = 1e-4

  /**
   * Trains an ANN.
   * Uses default convergence tolerance 1e-4 for LBFGS.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param batchSize
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param initialWeights
   * @param maxNumIterations specifies maximum number of training iterations.
   * @param convergenceTol
   * @return ANN model.
   */

  def train(trainingRDD: RDD[(Vector, Vector)],
    batchSize: Int,
    hiddenLayersTopology: Array[Int],
    initialWeights: Vector,
    maxNumIterations: Int,
    convergenceTol: Double): ANNmodel = {
    val topology = convertTopology(trainingRDD, hiddenLayersTopology)
    new ANN(topology, maxNumIterations, convergenceTol, batchSize).
      run(trainingRDD, initialWeights)
  }
  /**
   * Trains an ANN.
   * Uses default convergence tolerance 1e-4 for LBFGS.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param batchSize
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param maxNumIterations specifies maximum number of training iterations.
   * @return ANN model.
   */
  def train(trainingRDD: RDD[(Vector, Vector)],
    batchSize: Int,
    hiddenLayersTopology: Array[Int],
    maxNumIterations: Int): ANNmodel = {
    val topology = convertTopology(trainingRDD, hiddenLayersTopology)
    new ANN(topology, maxNumIterations, defaultTolerance, batchSize).
      run(trainingRDD, randomWeights(topology, false))
  }
  /**
   * Trains an ANN.
   * Uses default convergence tolerance 1e-4 for LBFGS.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param maxNumIterations specifies maximum number of training iterations.
   * @return ANN model.
   */
  def train(trainingRDD: RDD[(Vector, Vector)],
    hiddenLayersTopology: Array[Int],
    maxNumIterations: Int): ANNmodel = {
    train(trainingRDD, hiddenLayersTopology, maxNumIterations, defaultTolerance)
  }

  /**
   * Trains an ANN with given initial weights.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param initialWeights initial weights vector.
   * @param maxNumIterations maximum number of training iterations.
   * @param convergenceTol convergence tolerance for LBFGS. Smaller value for closer convergence.
   * @return ANN model.
   */
  def train(trainingRDD: RDD[(Vector, Vector)],
    hiddenLayersTopology: Array[Int],
    initialWeights: Vector,
    maxNumIterations: Int,
    convergenceTol: Double): ANNmodel = {
    val topology = convertTopology(trainingRDD, hiddenLayersTopology)
    new ANN(topology, maxNumIterations, convergenceTol).
      run(trainingRDD, initialWeights)
  }

  /**
   * Continues training of an ANN using customized convergence tolerance.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param maxNumIterations maximum number of training iterations.
   * @param convergenceTol convergence tolerance for LBFGS. Smaller value for closer convergence.
   * @return ANN model.
   */
  def train(trainingRDD: RDD[(Vector, Vector)],
    hiddenLayersTopology: Array[Int],
    maxNumIterations: Int,
    convergenceTol: Double): ANNmodel = {
    val topology = convertTopology(trainingRDD, hiddenLayersTopology)
    new ANN(topology, maxNumIterations, convergenceTol).
      run(trainingRDD, randomWeights(topology, false))
  }

  /**
   * Trains an ANN using customized convergence tolerance.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param model model of an already partly trained ANN.
   * @param maxNumIterations maximum number of training iterations.
   * @param convergenceTol convergence tolerance for LBFGS. Smaller value for closer convergence.
   * @return ANN model.
   */
  def train(trainingRDD: RDD[(Vector, Vector)],
    model: ANNmodel,
    maxNumIterations: Int,
    convergenceTol: Double): ANNmodel = {
    new ANN(model.topology, maxNumIterations, convergenceTol).
      run(trainingRDD, model.weights)
  }

  /**
   * Trains an ANN with given initial weights.
   * Uses default convergence tolerance 1e-4 for LBFGS.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param initialWeights initial weights vector.
   * @param maxNumIterations maximum number of training iterations.
   * @return ANN model.
   */
  def train(trainingRDD: RDD[(Vector, Vector)],
    hiddenLayersTopology: Array[Int],
    initialWeights: Vector,
    maxNumIterations: Int): ANNmodel = {
    train(trainingRDD, hiddenLayersTopology, initialWeights, maxNumIterations, defaultTolerance)
  }
  /**
   * Continues training of an ANN.
   * Uses default convergence tolerance 1e-4 for LBFGS.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param model model of an already partly trained ANN.
   * @param maxNumIterations maximum number of training iterations.
   * @return ANN model.
   */
  def train(trainingRDD: RDD[(Vector, Vector)],
    model: ANNmodel,
    maxNumIterations: Int): ANNmodel = {
    train(trainingRDD, model, maxNumIterations, defaultTolerance)
  }

  private def convertTopology(input: RDD[(Vector, Vector)],
    hiddenLayersTopology: Array[Int]): Array[Int] = {
    val instance = input.first
    instance._1.size +: hiddenLayersTopology :+ instance._2.size
  }

  /**
   * Provides a random weights vector.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @return random weights vector.
   */
  def randomWeights(trainingRDD: RDD[(Vector, Vector)],
    hiddenLayersTopology: Array[Int]): Vector = {
    val topology = convertTopology(trainingRDD, hiddenLayersTopology)
    return randomWeights(topology, false)
  }
  /**
   * Provides a random weights vector, using given random seed.
   *
   * @param trainingRDD RDD containing (input, output) pairs for later training.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param seed random generator seed.
   * @return random weights vector.
   */
  def randomWeights(trainingRDD: RDD[(Vector, Vector)],
    hiddenLayersTopology: Array[Int],
    seed: Int): Vector = {
    val topology = convertTopology(trainingRDD, hiddenLayersTopology)
    return randomWeights(topology, true, seed)
  }
  /**
   * Provides a random weights vector, using given random seed.
   *
   * @param inputLayerSize size of input layer.
   * @param outputLayerSize size of output layer.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param seed random generator seed.
   * @return random weights vector.
   */
  def randomWeights(inputLayerSize: Int,
    outputLayerSize: Int,
    hiddenLayersTopology: Array[Int],
    seed: Int): Vector = {
    val topology = inputLayerSize +: hiddenLayersTopology :+ outputLayerSize
    return randomWeights(topology, true, seed)
  }
  /**
   * @param topology number of nodes in every layer.
   * @param useSeed boolean to use seed
   * @param seed random generator seed
   * @return random weights vector
   */
  private def randomWeights(topology: Array[Int], useSeed: Boolean, seed: Int = 0): Vector = {
    val rand: XORShiftRandom =
      if (useSeed == false) new XORShiftRandom() else new XORShiftRandom(seed)

    var i: Int = 0
    var l: Int = 0
    val noWeights = {
      var tmp = 0
      var i = 1
      while (i < topology.size) {
        tmp = tmp + topology(i) * (topology(i - 1) + 1)
        i += 1
      }
      tmp
    }
    val initialWeightsArr = new Array[Double](noWeights)

    var pos = 0
    l = 1
    while (l < topology.length) {
      i = 0
      while (i < (topology(l) * (topology(l - 1) + 1))) {
        initialWeightsArr(pos) = (rand.nextDouble * 4.8 - 2.4) / (topology(l - 1) + 1)
        pos += 1
        i += 1
      }
      l += 1
    }

    Vectors.dense(initialWeightsArr)

  }

}