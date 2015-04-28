package org.template.classification

import io.prediction.controller.PPreparator

import grizzled.slf4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import math.{cos, Pi, sin}

import java.util.Calendar

class PreparedData(
  val data: RDD[LabeledPoint]
) extends Serializable

class Preparator extends PPreparator[TrainingData, PreparedData] {

  @transient lazy val logger = Logger[this.type]

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    new PreparedData(trainingData.data map {p =>
      LabeledPoint(p.energy_consumption, Preparator.toFeaturesVector(p.circuit_id, p.timestamp))
    })
  }
}

object Preparator {

  @transient lazy val logger = Logger[this.type]

  def toFeaturesVector(circuit_id: Int, timestamp: Long): Vector = {
    toFeaturesVector(circuit_id, timestamp, dummyCoding)
  }

  def toFeaturesVector(circuit_id: Int, timestamp: Long, coding: (Int, Int) => Array[Double]): Vector = {
    val cal = Calendar.getInstance()
    cal.setTimeInMillis(timestamp * 1000)
    val (maxHour, maxDayWeek, maxDayMonth, maxMonth) = (23, 6, 30, 11)
    val (hour, dayWeek, dayMonth, month) =
      (cal.get(Calendar.HOUR_OF_DAY),
        cal.get(Calendar.DAY_OF_WEEK) - 1,
        cal.get(Calendar.DAY_OF_MONTH) - 1,
        cal.get(Calendar.MONTH))

    val (hourC, hourCSquare, dayWeekC, dayMonthC, monthC, hourWeekC) =
      (coding(hour, maxHour),
        coding(hour * hour, maxHour * maxHour),
        coding(dayWeek, maxDayWeek),
        coding(dayMonth, maxDayMonth),
        coding(month, maxMonth),
        coding((hour + 1) * (dayWeek + 1) - 1, (maxHour + 1) * (maxDayWeek + 1) - 1))

    val features = hourC ++
      hourCSquare ++
      dayWeekC ++
      monthC ++
      dayMonthC ++
      hourWeekC

    Vectors.dense(features)
  }

  def circleCoding(value: Int, maxValue: Int): Array[Double] = {
    Array[Double](cos(2 * Pi * (value.toDouble / maxValue.toDouble)),
      sin(2 * Pi * (value.toDouble / maxValue.toDouble)))
  }

  def dummyCoding(value: Int, maxValue: Int): Array[Double] = {
    val arr = Array.fill[Double](maxValue)(0.0)
    if (value != 0) arr(value - 1) = 1.0
    arr
  }
}
