package org.template.classification

import io.prediction.controller.PPreparator

import grizzled.slf4j.Logger
import org.apache.spark.SparkContext
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet

class PreparedData(
  val dataSet: DataSet
) extends Serializable

class Preparator extends PPreparator[TrainingData, PreparedData] {

  @transient lazy val logger = Logger[this.type]

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    this.logger.info("START PREPARE")
    var dFeatures: List[Array[Double]] = List()
    var dLabels: List[Double] = List()

    for (lb <- trainingData.labeledPoints.collect()) {
      dFeatures = lb.features.toArray :: dFeatures
      dLabels = lb.label :: dLabels
    }

    logger.info(dFeatures.mkString(" "))
    logger.info(dLabels.mkString("\n"))

    val dsFeatures = Nd4j.create(dFeatures.reverse.toArray)
    val dsLabels = Nd4j.create(dLabels.reverse.toArray).transpose()
    this.logger.info("END PREPARE-")
    new PreparedData(new DataSet(dsFeatures, dsLabels))
  }
}
