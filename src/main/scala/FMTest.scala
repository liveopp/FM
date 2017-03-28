/**
  * Created by jimmyzhang on 2016/9/26.
  */

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SparkSession, DataFrame, Dataset}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

object FMTest {
  def oneHotEncoder(dataset: Dataset[_], features: Array[String], featureThresh: Int = 1000): DataFrame = {
    val schema = dataset.schema
    val labelMap = features.map { name => {
      val column = dataset.filter(col(name).isNotNull).select(name)
      val rdd = column.rdd
      val rddString: RDD[String] = schema(name).dataType match {
        case t: ArrayType if t.elementType == LongType => rdd.flatMap(_.getSeq[Long](0).map(_.toString))
        case t: ArrayType if t.elementType == IntegerType => rdd.flatMap(_.getSeq[Int](0).map(_.toString))
        case t: ArrayType if t.elementType == StringType => rdd.flatMap(_.getSeq[String](0))
        case _: IntegerType => rdd.map(_.getInt(0).toString)
        case _: LongType => rdd.map(_.getLong(0).toString)
        case _: StringType => rdd.map(_.getString(0))
      }
      val counts = rddString.countByValue
      var numFeature = 0
      val labels = counts mapValues { cnt =>
        if (cnt < featureThresh) 0
        else {
          numFeature += 1
          numFeature
        }
      } map identity
      (name, (numFeature+1, labels))
    }}.toMap

    val numMap = labelMap.mapValues(_._1)
    val valueMap = labelMap.mapValues(_._2.toMap)
//    spark.createDataFrame(List(FeatureMap(numMap, valueMap)))
//      .repartition(1).write.mode("overwrite")
//      .parquet(s"/data/cupid_algo/cpc/ctr/feature_map/dt=${bizDate}")

    val spark = dataset.sparkSession
    val bcValueMap = spark.sparkContext.broadcast(valueMap)
    val bcNumMap = spark.sparkContext.broadcast(numMap)

    val encoder = (name: String) => udf[Vector, Any]( ceil => {
      val numFeature = bcNumMap.value(name)
      val labels = bcValueMap.value(name)
      ceil match {
        case null => Vectors.zeros(numFeature)
        case ceil: Seq[Any] => 
          val idx = ceil.map(c => labels(c.toString)).toArray.sorted.distinct
          Vectors.sparse(numFeature, idx, Array.fill[Double](idx.length)(1.0))
        case ceil: Any =>
          Vectors.sparse(numFeature, Array(labels(ceil.toString)), Array(1.0))
      }
    })
    val oneHotColArry = features.map(f => encoder(f)(col(f)).as(f)) :+ col("label").cast(DoubleType)
    dataset.select(oneHotColArry: _*)
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("FM test")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate
    import spark.implicits._
    val dataset = spark.read.parquet("data")
    val pos = dataset.filter("label = 1")
    val neg = dataset.filter("label = 0").sample(false, 0.1)
    val sampled = pos.union(neg).persist
    val onehotDF = oneHotEncoder(sampled, sampled.schema.fieldNames
    val assembler = new VectorAssembler()
      .setInputCols(onehotDF.schema.fieldNames.filter(_ != "label"))
      .setOutputCol("features")
    val output = assembler.transform(onehotDF).select("features", "label")
    val Array(trainingData, testData) = output.randomSplit(Array(0.7, 0.3))

    val fm = new FM()
      .setMaxIter(50)

    val fmModel = fm.fit(trainingData)
    println(s"Coefficients: ${fmModel.coefficients} Intercept: ${fmModel.intercept}")


	val trainingSummary = fmModel.summary

	// Obtain the objective per iteration.
	val objectiveHistory = trainingSummary.objectiveHistory
	objectiveHistory.foreach(loss => println(loss))
	// Obtain the metrics useful to judge performance on test data.
	// We cast the summary to a BinaryLogisticRegressionSummary since the problem is a
	// binary classification problem.
	val binarySummary = trainingSummary.asInstanceOf[BinaryFactorizationMachineSummary]

	// Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
	println(s"train auc:${binarySummary.areaUnderROC}")

    val predictions = fmModel.transform(testData)
    val testEvalue = new BinaryClassificationEvaluator()
    println(s"test auc: ${testEvalue.evaluate(predictions)}")

    spark.stop()
  }
}
