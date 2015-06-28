package detrevid.predictionio.loadforecasting

import io.prediction.controller.PPreparator
import io.prediction.controller.SanityCheck

import grizzled.slf4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import math.{cos, Pi, sin}

import java.util.{Calendar, TimeZone}

class PreparedData (
  val circuitsIds: Array[Int],
  val data: RDD[(Int, LabeledPoint)]
) extends Serializable with SanityCheck {

  override def sanityCheck(): Unit = {
    require(data.take(1).nonEmpty, s"data cannot be empty!")
  }
}

class Preparator extends PPreparator[TrainingData, PreparedData] {

  @transient lazy val logger = Logger[this.type]

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    val circuitsIds = trainingData.data map { _.circuit_id } distinct() collect()

    val data = trainingData.data map {
      ev => (ev.circuit_id, LabeledPoint(ev.energy_consumption,
        Preparator.toFeaturesVector(ev.circuit_id, ev.timestamp)))
    } cache()

    new PreparedData(circuitsIds, data)
  }
}

object Preparator {

  @transient lazy val logger = Logger[this.type]
  @transient lazy val timeZone = TimeZone.getTimeZone("America/Los_Angeles")

  def getLocalTime(timestamp: Long, timeZone: TimeZone): Calendar = {
    val timeInMs: Long = timestamp * 1000
    val cal = Calendar.getInstance()
    val utcTimeInMs: Long = timeInMs - cal.getTimeZone.getOffset(timeInMs)
    val localTimeInMs: Long = utcTimeInMs + timeZone.getOffset(utcTimeInMs)
    cal.setTimeInMillis(localTimeInMs)
    cal
  }

  def toFeaturesVector(circuit_id: Int, timestamp: Long): Vector = {
    toFeaturesVector(circuit_id, timestamp, dummyCoding)
  }

  def toFeaturesVector(circuit_id: Int, timestamp: Long,
                       coding: (Int, Int) => Array[Double]): Vector = {
    val cal = getLocalTime(timestamp, timeZone)

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
