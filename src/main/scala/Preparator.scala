package org.template.classification

import io.prediction.controller.PPreparator

import grizzled.slf4j.Logger
import org.apache.spark.SparkContext
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet

import java.util.Calendar

class PreparedData(
  val dataSet: DataSet
) extends Serializable

class Preparator extends PPreparator[TrainingData, PreparedData] {

  @transient lazy val logger = Logger[this.type]

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    this.logger.info("START PREPARE") //Debug
    var dFeatures: List[Array[Double]] = List()
    var dLabels: List[Double] = List()

    for (lb <- trainingData.data.collect()) {
      dFeatures = Preparator.toFeaturesArray(lb.circuit_id, lb.timestamp) :: dFeatures
      dLabels = lb.energy_consumption :: dLabels
    }

    logger.info(dFeatures.mkString(" ")) //Debug
    logger.info(dLabels.mkString("\n")) //Debug

    val dsFeatures = Nd4j.create(dFeatures.reverse.toArray)
    val dsLabels = Nd4j.create(dLabels.reverse.toArray).transpose()
    this.logger.info("END PREPARE") //Debug
    new PreparedData(new DataSet(dsFeatures, dsLabels))
  }
}

object Preparator {
  def getFeaturesArraySize: Int = 4

  def toFeaturesArray(circuit_id: Int, timestamp: Int): Array[Double] = {
    val calendar = Calendar.getInstance()
    calendar.setTimeInMillis(timestamp * 1000)

    Array[Double](circuit_id,
                  calendar.get(Calendar.HOUR_OF_DAY),
                  calendar.get(Calendar.DAY_OF_WEEK),
                  calendar.get(Calendar.DAY_OF_MONTH)
                  )
  }
}
