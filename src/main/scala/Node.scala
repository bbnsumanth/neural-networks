import scala.math._

class Node {
  //sigmoid mathod is defined
  def sigmoid(x:Double):Double = {
      (1/(1+exp(-x)))
    }
   
  def output(input:Double):Double={
    sigmoid(input)
    
    }

}
