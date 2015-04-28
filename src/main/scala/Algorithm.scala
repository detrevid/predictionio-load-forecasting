package detrevid.predictionio.energyforecasting

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import grizzled.slf4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.{LinearRegressionModel, LinearRegressionWithSGD}

case class AlgorithmParams(
  iterations: Int    = 10000,
  stepSize:   Double = 0.1
) extends Params

class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Model = {
    require(data.data.take(1).nonEmpty,
      s"RDD[labeledPoints] in PreparedData cannot be empty." +
        " Please check if DataSource generates TrainingData" +
        " and Preparator generates PreparedData correctly.")

    val lin = new LinearRegressionWithSGD()
    lin.setIntercept(true)
    lin.optimizer
      .setNumIterations(ap.iterations)
      .setStepSize(ap.stepSize)

    new Model(lin.run(data.data))
  }

  def predict(model: Model, query: Query): PredictedResult = {
    val label : Double = model.predict(query)
    new PredictedResult(label)
  }
}

class Model(val mod: LinearRegressionModel) extends Serializable {

  @transient lazy val logger = Logger[this.type]

  def predict(query: Query): Double = {
    val features = Preparator.toFeaturesVector(query.circuit_id, query.timestamp)
    mod.predict(features)
  }
}