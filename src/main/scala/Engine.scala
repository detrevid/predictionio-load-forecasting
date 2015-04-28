package detrevid.predictionio.energyforecasting

import io.prediction.controller.{Engine, EngineFactory}

class Query(
  val circuit_id: Int,
  val timestamp: Long
) extends Serializable

class PredictedResult(
  val label: Double
) extends Serializable

class ActualResult(
  val label: Double
) extends Serializable

object ClassificationEngine extends EngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("alg" -> classOf[Algorithm]),
      classOf[Serving])
  }
}
