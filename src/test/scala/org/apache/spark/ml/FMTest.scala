/**
  * Created by jimmyzhang on 2016/9/26.
  */

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.apache.spark.ml.classification.FactorizationMachine

object FMTest {
  case class instense(feature: Vector, weight: Double, label: Double)

  def main(args: Array[String]): Unit = {
   val spark = SparkSession.builder()
      .master("local")
      .appName("FM test")
      .getOrCreate
    val dataset = spark.createDataFrame(
      instense(Vectors.dense(1, 0, 1.0), 1, 1) ::
        instense(Vectors.dense(0, 1, 0), 1, 0) ::
        instense(Vectors.dense(0, 1, 1), 1, 0) ::
        instense(Vectors.dense(1, 1, 1), 1, 1) ::
        Nil
    )
    val fm = new FactorizationMachine(2)
      .setMaxIter(10)
      .setRegParam(0.3)
    val fmModel = fm.fit(dataset)
    println(s"Coefficients: ${fmModel.coefficients} Intercept: ${fmModel.intercept}")
    spark.stop
  }
}
