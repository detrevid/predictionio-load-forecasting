package org.template.classification

import io.prediction.controller.PPreparator

import grizzled.slf4j.Logger
import org.apache.spark.SparkContext
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet

import math.{cos, Pi, sin}

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

    //logger.info(dFeatures.mkString(" ")) //Debug
    //logger.info(dLabels.mkString("\n")) //Debug

    val dsFeatures = Nd4j.create(dFeatures.reverse.toArray)
    val dsLabels = Nd4j.create(dLabels.reverse.toArray).transpose()
    this.logger.info("END PREPARE") //Debug
    new PreparedData(new DataSet(dsFeatures, dsLabels))
  }
}

object Preparator {
  @transient lazy val logger = Logger[this.type]

  def getFeaturesArraySize: Int = 9

  def toFeaturesArray(circuit_id: Int, timestamp: Long): Array[Double] = {
    val cal = Calendar.getInstance()
    cal.setTimeInMillis(timestamp * 1000)
    val (maxHour, maxDayWeek, maxDayMonth) =
      (cal.getActualMaximum(Calendar.HOUR_OF_DAY),
        cal.getActualMaximum(Calendar.DAY_OF_WEEK),
        cal.getActualMaximum(Calendar.DAY_OF_MONTH))
    val (hour, dayWeek, dayMonth) =
      (cal.get(Calendar.HOUR_OF_DAY),
        cal.get(Calendar.DAY_OF_WEEK) - 1,
        cal.get(Calendar.DAY_OF_MONTH) - 1)

    val (hourC, dayWeekC, dayMonthC, hourWeekC) =
      (circle_data(hour, maxHour),
        circle_data(dayWeek, maxDayWeek),
        circle_data(dayMonth, maxDayMonth),
        circle_data(hour * dayWeek, maxHour * maxDayWeek))

    Array[Double](circuit_id,
      hourC._1, hourC._2,
      dayWeekC._1, dayWeekC._2,
      dayMonthC._1, dayMonthC._2,
      hourWeekC._1, hourWeekC._2
    )
  }

  def circle_data(value: Double, max_value: Double): (Double, Double) = {
    (cos(2 * Pi * (value / max_value)),
      sin(2 * Pi * (value / max_value)))
  }
}
