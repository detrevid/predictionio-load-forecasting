package detrevid.predictionio.loadforecasting

import io.prediction.controller.AverageMetric
import io.prediction.controller.EmptyEvaluationInfo
import io.prediction.controller.EngineParams
import io.prediction.controller.EngineParamsGenerator
import io.prediction.controller.Evaluation

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import math.{pow, sqrt}

case class RMSEMetric()
  extends AverageMetric[EmptyEvaluationInfo, Query, PredictedResult, ActualResult] {

  override
  def calculate(sc: SparkContext,
                evalDataSet: Seq[(EmptyEvaluationInfo,
                  RDD[(Query, PredictedResult, ActualResult)])]): Double = {
    sqrt(super.calculate(sc, evalDataSet))
  }

  def calculate(query: Query, predicted: PredictedResult, actual: ActualResult): Double =
    pow(predicted.label - actual.label, 2)

  override
  def compare(r0: Double, r1: Double): scala.Int = {
    -1 * super.compare(r0, r1)
  }
}

object RMSEEvaluation extends Evaluation {
  engineMetric = (ForecastingEngine(), new RMSEMetric())
}

object EngineParamsList extends EngineParamsGenerator {

  private[this] val baseEP = EngineParams(
    dataSourceParams = DataSourceParams(appName = "EnergyForecaster", evalK = Some(5)))

  engineParamsList = Seq(
    baseEP.copy(
      algorithmParamsList = Seq(
        ("alg", AlgorithmParams(iterations = 100, stepSize = 0.01))
      )),
    baseEP.copy(
      algorithmParamsList = Seq(
        ("alg", AlgorithmParams(iterations = 100, stepSize = 0.1))
      )),
    baseEP.copy(
      algorithmParamsList = Seq(
        ("alg", AlgorithmParams(iterations = 1000, stepSize = 0.01))
      )),
    baseEP.copy(
      algorithmParamsList = Seq(
        ("alg", AlgorithmParams(iterations = 1000, stepSize = 0.1))
      ))
  )
}

