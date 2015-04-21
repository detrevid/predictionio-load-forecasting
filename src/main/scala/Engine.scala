package org.template.classification

import io.prediction.controller.EngineFactory
import io.prediction.controller.Engine

class Query(
  val circuit_id: Int,
  val timestamp: Int
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
