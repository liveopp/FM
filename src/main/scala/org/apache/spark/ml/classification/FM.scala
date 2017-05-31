/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.classification

import scala.collection.mutable
import scala.util.Random
import breeze.linalg.{*, sum, DenseVector => BDV}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS, OWLQN => BreezeOWLQN}
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkException
import org.apache.spark.annotation.Experimental
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.linalg.BLAS._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.storage.StorageLevel

/**
  * Params for factorization machine.
  */
private[classification] trait FMParams extends ProbabilisticClassifierParams
  with HasMaxIter with HasFitIntercept with HasTol with HasWeightCol with HasThreshold {

  /**
    * Param for rank of the factorization (positive).
    * @group param
    */
  val rank = new IntParam(this, "rank", "rank of the factorization", ParamValidators.gtEq(1))

  /** @group getParam */
  def getRank: Int = $(rank)

  /**
    * Param for L1 regularization of first-order weights (positive).
    * @group param
    */
  val regParamL1 = new DoubleParam(this, "regParamL1",
    "L1-regularization parameter for first-order weights", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegParamL1: Double = $(regParamL1)

  /**
    * Param for features column beginIndex for L1 regularization.
    * @group param
    */
  val regParamL1Index = new IntParam(this, "regParamL1Index",
    "features column beginIndex for L1-regularization", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegParamL1Index: Int = $(regParamL1Index)

  /**
    * Param for L2 regularization of first-order weights (positive).
    * @group param
    */
  val regParamFirst = new DoubleParam(this, "regParamFirst",
    "L2-regularization parameter for first-order weights", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegParamFirst: Double = $(regParamFirst)

  /**
    * Param for L2 regularization of second-order factors (positive).
    * @group param
    */
  val regParamSecond = new DoubleParam(this, "regParamSecond",
    "L2 regularization parameter for second-order factors", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegParamSecond: Double = $(regParamSecond)

  /**
    * Set threshold in binary classification, in range [0, 1].
    *
    * If the estimated probability of class label 1 is > threshold, then predict 1, else 0.
    * A high threshold encourages the model to predict 0 more often;
    * a low threshold encourages the model to predict 1 more often.
    *
    * Default is 0.5.
    * @group setParam
    */
  def setThreshold(value: Double): this.type = set(threshold, value)

  /**
    * Get threshold for binary classification.
    *
    * @group getParam
    * @throws IllegalArgumentException if [[thresholds]] is set to an array of length other than 2.
    */
  override def getThreshold: Double = $(threshold)
}

/**
  * Factorization machines.
  * Currently, this class only supports binary classification.
  */
class FM(override val uid: String)
  extends ProbabilisticClassifier[Vector, FM, FMModel]
    with FMParams with DefaultParamsWritable with Logging {

  def this() = this(Identifiable.randomUID("fm"))

  private var optInitialModel: Option[FMModel] = None
  /** @group setParam */
  private[spark] def setInitialModel(model: FMModel): this.type = {
    this.optInitialModel = Some(model)
    this
  }

  /**
    * Set the number of factors.
    * Default is 10.
    * @group setParam
    */
  def setRank(value: Int): this.type = set(rank, value)
  setDefault(rank -> 10)

  /**
    * Set the L1-regularization parameter of first-order weights.
    * Default is 0.0.
    * @group setParam
    */
  def setRegParamL1(value: Double): this.type = set(regParamL1, value)
  setDefault(regParamL1 -> 0.0)

  /**
    * Set the L1-regularization parameter of features column beginIndex.
    * Default is 0.
    * @group setParam
    */
  def setRegParamL1Index(value: Int): this.type = set(regParamL1Index, value)
  setDefault(regParamL1Index -> 0)

  /**
    * Set the L2-regularization parameter of first-order weights.
    * Default is 0.0.
    * @group setParam
    */
  def setRegParamFirst(value: Double): this.type = set(regParamFirst, value)
  setDefault(regParamFirst -> 0.0)

  /**
    * Set the L2-regularization parameter of second-order factors.
    * Default is 0.0.
    * @group setParam
    */
  def setRegParamSecond(value: Double): this.type = set(regParamSecond, value)
  setDefault(regParamSecond -> 0.0)

  /**
    * Set the maximum number of iterations.
    * Default is 100.
    * @group setParam
    */
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  /**
    * Set the convergence tolerance of iterations.
    * Smaller value will lead to higher accuracy with the cost of more iterations.
    * Default is 1E-6.
    * @group setParam
    */
  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  /**
    * Whether to fit an intercept term.
    * Default is true.
    * @group setParam
    */
  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)
  setDefault(fitIntercept -> true)

  override def setThreshold(value: Double): this.type = super.setThreshold(value)

  override def getThreshold: Double = super.getThreshold

  /**
    * Whether to over-/under-sample training instances according to the given weights in weightCol.
    * If not set or empty String, all instances are treated equally (weight 1.0).
    * Default is not set, so all instances have weight one.
    * @group setParam
    */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  override protected[spark] def train(dataset: Dataset[_]): FMModel = {
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    train(dataset, handlePersistence)
  }

  protected[spark] def train(dataset: Dataset[_], handlePersistence: Boolean):
  FMModel = {
    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val instances: RDD[Instance] =
      dataset.select(col($(labelCol)).cast(DoubleType), w, col($(featuresCol))).rdd.map {
        case Row(label: Double, weight: Double, features: Vector) =>
          Instance(label, weight, features)
      }

    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val instr = Instrumentation.create(this, instances)
    instr.logParams(rank, regParamFirst, regParamSecond, threshold,
      maxIter, tol, fitIntercept)

    val labelSummarizer = {
      val seqOp = (c: MultiClassSummarizer,
                   instance: Instance) => c.add(instance.label, instance.weight)

      val combOp = (c1: MultiClassSummarizer, c2: MultiClassSummarizer) => c1.merge(c2)

      instances.treeAggregate(new MultiClassSummarizer)(seqOp, combOp)
    }

    val histogram = labelSummarizer.histogram
    val numInvalid = labelSummarizer.countInvalid
    val numClasses = histogram.length
    val numFeatures = instances.first.features.size
    val numParams = numFeatures * ($(rank) + 1)

    instr.logNumClasses(numClasses)
    instr.logNumFeatures(numFeatures)

    val (coefficients, intercept, objectiveHistory) = {
      if (numInvalid != 0) {
        val msg = s"Classification labels should be in {0 to ${numClasses - 1}" +
          s"Found $numInvalid invalid labels."
        logError(msg)
        throw new SparkException(msg)
      }

      if (numClasses > 2) {
        val msg = s"Currently, FM only supports binary classification." +
          s" Found $numClasses in the input dataset."
        logError(msg)
        throw new SparkException(msg)
      } else if ($(fitIntercept) && numClasses == 2 && histogram(0) == 0.0) {
        logWarning(s"All labels are one and fitIntercept=true, so the coefficients will be " +
          s"zeros and the intercept will be positive infinity; as a result, " +
          s"training is not needed.")
        (Vectors.sparse(numFeatures, Seq()), Double.PositiveInfinity, Array.empty[Double])
      } else if ($(fitIntercept) && numClasses == 1) {
        logWarning(s"All labels are zero and fitIntercept=true, so the coefficients will be " +
          s"zeros and the intercept will be negative infinity; as a result, " +
          s"training is not needed.")
        (Vectors.sparse(numFeatures, Seq()), Double.NegativeInfinity, Array.empty[Double])
      } else {
        if (!$(fitIntercept) && numClasses == 2 && histogram(0) == 0.0) {
          logWarning(s"All labels are one and fitIntercept=false. It's a dangerous ground, " +
            s"so the algorithm may not converge.")
        } else if (!$(fitIntercept) && numClasses == 1) {
          logWarning(s"All labels are zero and fitIntercept=false. It's a dangerous ground, " +
            s"so the algorithm may not converge.")
        }

        val initialCoefficientsArray = Array.ofDim[Double](numParams + {if ($(fitIntercept)) 1 else 0})

        // model update
        if (optInitialModel.isDefined && optInitialModel.get.coefficientMatrix.numActives != numParams) {
          val vecSize = optInitialModel.get.coefficientMatrix.numActives
          logWarning(
            s"Initial coefficients will be ignored!! As its size $vecSize did not match the " +
              s"expected size $numParams")
        }
        if (optInitialModel.isDefined && optInitialModel.get.coefficientMatrix.numActives == numParams) {
          optInitialModel.get.coefficientMatrix.foreachActive { case (row, col, value) =>
            val index = row * numFeatures + col
            initialCoefficientsArray(index) = value
          }
          if ($(fitIntercept)) {
            initialCoefficientsArray(numParams) == optInitialModel.get.intercept
          }
        } else {
          // set up the initial values of factors
          var i = numFeatures
          while (i < numFeatures * ($(rank) + 1)) {
            initialCoefficientsArray(i) = math.sqrt($(rank)) * (Random.nextDouble - 0.5) / 50
            i += 1
          }
          if ($(fitIntercept)) {
            /*
               For binary logistic regression, when we initialize the coefficients as zeros,
               it will converge faster if we initialize the intercept such that
               it follows the distribution of the labels.

               {{{
                 P(0) = 1 / (1 + \exp(b)), and
                 P(1) = \exp(b) / (1 + \exp(b))
               }}}, hence
               {{{
                 b = \log{P(1) / P(0)} = \log{count_1 / count_0}
               }}}
             */
            initialCoefficientsArray(numParams) = math.log(histogram(1) / histogram(0))
          }
        }


        /*
            Use L-BFGS for L2 regularization, OWL-QN for L1 regularization.
         */

        val initialCoefficientsWithIntercept = BDV(initialCoefficientsArray)
        val costFun = new FactorCostFun(instances, numFeatures, $(rank), $(fitIntercept),
          $(regParamFirst), $(regParamSecond), $(regParamL1) > 0.0)

        val optimizer = if ($(regParamL1) == 0.0) {
          new BreezeLBFGS[BDV[Double]]($(maxIter), 10, $(tol))
        } else {
          val l1reg = (index: Int) => {
            if (index < $(regParamL1Index) && index < numFeatures) {
              $(regParamL1)
            } else {
              0
            }
          }
          new BreezeOWLQN[Int, BDV[Double]]($(maxIter), 10, l1reg, $(tol))
        }
        val states = optimizer.iterations(new CachedDiffFunction(costFun),
          initialCoefficientsWithIntercept)

        /*
         Note that in Factorization Machine, the objective history (loss + regularization)
         is log-likelihood which is invariance under feature standardization. As a result,
         the objective history from optimizer is the same as the one in the original space.
       */
        val arrayBuilder = mutable.ArrayBuilder.make[Double]
        var state: optimizer.State = null
        while (states.hasNext) {
          state = states.next()
          arrayBuilder += state.adjustedValue
        }

        if (state == null) {
          val msg = s"${optimizer.getClass.getName} failed."
          logError(msg)
          throw new SparkException(msg)
        }

        /*
         The coefficients are trained in the scaled space; we're converting them back to
         the original space.
         Note that the intercept in scaled space and original space is the same;
         as a result, no scaling is needed.
       */
        val rawCoefficients = if ($(regParamL1) == 0.0) {
          state.x.toArray.clone()
        } else {
          FMSparse.makeSparse(state.x, numFeatures, $(rank)).clone()
        }

        if ($(fitIntercept)) {
          (Vectors.dense(rawCoefficients.dropRight(1)).compressed, rawCoefficients.last,
            arrayBuilder.result())
        } else {
          (Vectors.dense(rawCoefficients).compressed, 0.0, arrayBuilder.result())
        }
      }
    }

    if (handlePersistence) instances.unpersist()

    val model = copyValues(new FMModel(uid,
      Matrices.dense(numFeatures, $(rank)+1, coefficients.toArray),
      intercept))
    val (summaryModel, probabilityColName) = model.findSummaryModelAndProbabilityCol()
    val fmSummary = new BinaryFMTrainingSummary(
      summaryModel.transform(dataset),
      probabilityColName,
      $(labelCol),
      $(featuresCol),
      objectiveHistory)
    val m = model.setSummary(fmSummary)
    instr.logSuccess(m)
    m
  }

  override def copy(extra: ParamMap): FM = defaultCopy(extra)
}

object FM extends DefaultParamsReadable[FM] {

  override def load(path: String): FM = super.load(path)
}

/**
  * Model produced by [[FM]].
  */
class FMModel private[spark](
                              override val uid: String,
                              val coefficientMatrix: Matrix,
                              val intercept: Double)
  extends ProbabilisticClassificationModel[Vector, FMModel]
    with FMParams with MLWritable {

  override def setThreshold(value: Double): this.type = super.setThreshold(value)

  override def getThreshold: Double = super.getThreshold

  set(rank, coefficientMatrix.numCols - 1)

  override val numFeatures: Int = coefficientMatrix.numRows

  override val numClasses: Int = 2

  private val factorNorms: Vector = {
    val breezeMatrix = coefficientMatrix.asBreeze.toDenseMatrix
    val factorMatrix = breezeMatrix(::, 1 to -1)
    val squareMatrix = factorMatrix :* factorMatrix
    Vectors.fromBreeze(sum(squareMatrix(*, ::)))
  }

  /** Margin (rawPrediction) for class label 1.  For binary classification only. */
  private val margin: Vector => Double = (features) => {
    val factorSumArray = Array.ofDim[Double]($(rank))
    var sum = 0.0
    features.foreachActive { (index, value) =>
      if (value != 0.0) {
        sum += coefficientMatrix(index, 0) * value
        sum -= 0.5 * factorNorms(index) * value * value
        var iFactor = 0
        while (iFactor < $(rank)) {
          factorSumArray(iFactor) += coefficientMatrix(index, iFactor+1) * value
          iFactor += 1
        }
      }
    }
    sum += 0.5 * factorSumArray.map(f => f * f).sum
    sum + intercept
  }

  /** Score (probability) for class label 1. For binary classification only. */
  private val score: Vector => Double = (features) => {
    val m = margin(features)
    1.0 / (1.0 + math.exp(-m))
  }

  private var trainingSummary: Option[FMTrainingSummary] = None

  /**
    * Gets summary of model on training set. An exception is
    * thrown if `trainingSummary == None`.
    */
  def summary: FMTrainingSummary = trainingSummary.getOrElse {
    throw new SparkException("No training summary available for this LogisticRegressionModel")
  }

  /**
    * If the probability column is set returns the current model and probability column,
    * otherwise generates a new column and sets it as the probability column on a new copy
    * of the current model.
    */
  private[classification] def findSummaryModelAndProbabilityCol():
  (FMModel, String) = {
    $(probabilityCol) match {
      case "" =>
        val probabilityColName = "probability_" + java.util.UUID.randomUUID.toString
        (copy(ParamMap.empty).setProbabilityCol(probabilityColName), probabilityColName)
      case p => (this, p)
    }
  }

  private[classification] def setSummary(
                                          summary: FMTrainingSummary): this.type = {
    this.trainingSummary = Some(summary)
    this
  }

  /** Indicates whether a training summary exists for this model instance. */
  def hasSummary: Boolean = trainingSummary.isDefined

  /**
    * Evaluates the model on a test dataset.
    * @param dataset Test dataset to evaluate model on.
    */
  def evaluate(dataset: Dataset[_]): FMSummary = {
    // Handle possible missing or invalid prediction columns
    val (summaryModel, probabilityColName) = findSummaryModelAndProbabilityCol()
    new BinaryFMSummary(summaryModel.transform(dataset),
      probabilityColName, $(labelCol), $(featuresCol))
  }

  /**
    * Predict label for the given feature vector.
    * The behavior of this can be adjusted using [[thresholds]].
    */
  override protected def predict(features: Vector): Double = {
    // Note: We should use getThreshold instead of $(threshold) since getThreshold is overridden.
    if (score(features) > getThreshold) 1 else 0
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        var i = 0
        val size = dv.size
        while (i < size) {
          dv.values(i) = 1.0 / (1.0 + math.exp(-dv.values(i)))
          i += 1
        }
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in LogisticRegressionModel:" +
          " raw2probabilitiesInPlace encountered SparseVector")
    }
  }

  override protected def predictRaw(features: Vector): Vector = {
    val m = margin(features)
    Vectors.dense(-m, m)
  }

  override def copy(extra: ParamMap): FMModel = {
    val newModel = copyValues(new FMModel(uid, coefficientMatrix, intercept), extra)
    if (trainingSummary.isDefined) newModel.setSummary(trainingSummary.get)
    newModel.setParent(parent)
  }

  override protected def raw2prediction(rawPrediction: Vector): Double = {
    // Note: We should use getThreshold instead of $(threshold) since getThreshold is overridden.
    val t = getThreshold
    val rawThreshold = if (t == 0.0) {
      Double.NegativeInfinity
    } else if (t == 1.0) {
      Double.PositiveInfinity
    } else {
      math.log(t / (1.0 - t))
    }
    if (rawPrediction(1) > rawThreshold) 1 else 0
  }

  override protected def probability2prediction(probability: Vector): Double = {
    // Note: We should use getThreshold instead of $(threshold) since getThreshold is overridden.
    if (probability(1) > getThreshold) 1 else 0
  }

  /**
    * Returns a [[org.apache.spark.ml.util.MLWriter]] instance for this ML instance.
    *
    * For [[FMModel]], this does NOT currently save the training [[summary]].
    * An option to save [[summary]] may be added in the future.
    *
    * This also does not save the [[parent]] currently.
    */
  override def write: MLWriter = new FMModel.FMModelWriter(this)
}


object FMModel extends MLReadable[FMModel] {

  override def read: MLReader[FMModel] = new FMModelReader

  override def load(path: String): FMModel = super.load(path)

  /** [[MLWriter]] instance for [[FMModel]] */
  private[FMModel]
  class FMModelWriter(instance: FMModel)
    extends MLWriter with Logging {

    private case class Data(intercept: Double, coefficients: Matrix)

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      // Save model data: intercept, coefficients
      val data = Data(instance.intercept, instance.coefficientMatrix)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class FMModelReader extends MLReader[FMModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[FMModel].getName

    override def load(path: String): FMModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.format("parquet").load(dataPath)

      val model = {
        val Row(intercept: Double, coefficientMatrix: Matrix) = data.head()
        new FMModel(metadata.uid, coefficientMatrix, intercept)
      }

      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }
}

/**
  * Abstraction for multinomial Logistic Regression Training results.
  * Currently, the training summary ignores the training weights except
  * for the objective trace.
  */
sealed trait FMTrainingSummary extends FMSummary {

  /** objective function (scaled loss + regularization) at each iteration. */
  def objectiveHistory: Array[Double]

  /** Number of training iterations until termination */
  def totalIterations: Int = objectiveHistory.length

}

/**
  * Abstraction for Logistic Regression Results for a given model.
  */
sealed trait FMSummary extends Serializable {

  /** Dataframe output by the model's `transform` method. */
  def predictions: DataFrame

  /** Field in "predictions" which gives the probability of each class as a vector. */
  def probabilityCol: String

  /** Field in "predictions" which gives the true label of each instance (if available). */
  def labelCol: String

  /** Field in "predictions" which gives the features of each instance as a vector. */
  def featuresCol: String

}

/**
  * :: Experimental ::
  * Logistic regression training results.
  *
  * @param predictions dataframe output by the model's `transform` method.
  * @param probabilityCol field in "predictions" which gives the probability of
  *                       each class as a vector.
  * @param labelCol field in "predictions" which gives the true label of each instance.
  * @param featuresCol field in "predictions" which gives the features of each instance as a vector.
  * @param objectiveHistory objective function (scaled loss + regularization) at each iteration.
  */
class BinaryFMTrainingSummary private[classification](
                                                       predictions: DataFrame,
                                                       probabilityCol: String,
                                                       labelCol: String,
                                                       featuresCol: String,
                                                       val objectiveHistory: Array[Double])
  extends BinaryFMSummary(predictions, probabilityCol, labelCol, featuresCol)
    with FMTrainingSummary {

}

/**
  * :: Experimental ::
  * Binary Logistic regression results for a given model.
  *
  * @param predictions dataframe output by the model's `transform` method.
  * @param probabilityCol field in "predictions" which gives the probability of
  *                       each class as a vector.
  * @param labelCol field in "predictions" which gives the true label of each instance.
  * @param featuresCol field in "predictions" which gives the features of each instance as a vector.
  */
@Experimental
class BinaryFMSummary private[classification](
                                               @transient override val predictions: DataFrame,
                                               override val probabilityCol: String,
                                               override val labelCol: String,
                                               override val featuresCol: String) extends FMSummary {


  private val sparkSession = predictions.sparkSession
  import sparkSession.implicits._

  /**
    * Returns a BinaryClassificationMetrics object.
    */
  // TODO: Allow the user to vary the number of bins using a setBins method in
  // BinaryClassificationMetrics. For now the default is set to 100.
  @transient private val binaryMetrics = new BinaryClassificationMetrics(
    predictions.select(probabilityCol, labelCol).rdd.map {
      case Row(score: Vector, label: Double) => (score(1), label)
    }, 100
  )

  /**
    * Returns the receiver operating characteristic (ROC) curve,
    * which is a Dataframe having two fields (FPR, TPR)
    * with (0.0, 0.0) prepended and (1.0, 1.0) appended to it.
    * See http://en.wikipedia.org/wiki/Receiver_operating_characteristic
    *
    * Note: This ignores instance weights (setting all to 1.0) from `FM.weightCol`.
    *       This will change in later Spark versions.
    */
  @transient lazy val roc: DataFrame = binaryMetrics.roc().toDF("FPR", "TPR")

  /**
    * Computes the area under the receiver operating characteristic (ROC) curve.
    *
    * Note: This ignores instance weights (setting all to 1.0) from `FM.weightCol`.
    */
  lazy val areaUnderROC: Double = binaryMetrics.areaUnderROC()

  /**
    * Returns the precision-recall curve, which is a Dataframe containing
    * two fields recall, precision with (0.0, 1.0) prepended to it.
    *
    * Note: This ignores instance weights (setting all to 1.0) from `FM.weightCol`.
    */
  @transient lazy val pr: DataFrame = binaryMetrics.pr().toDF("recall", "precision")

  /**
    * Returns a dataframe with two fields (threshold, F-Measure) curve with beta = 1.0.
    *
    * Note: This ignores instance weights (setting all to 1.0) from `FM.weightCol`.
    */
  @transient lazy val fMeasureByThreshold: DataFrame = {
    binaryMetrics.fMeasureByThreshold().toDF("threshold", "F-Measure")
  }

  /**
    * Returns a dataframe with two fields (threshold, precision) curve.
    * Every possible probability obtained in transforming the dataset are used
    * as thresholds used in calculating the precision.
    *
    * Note: This ignores instance weights (setting all to 1.0) from `FM.weightCol`.
    */
  @transient lazy val precisionByThreshold: DataFrame = {
    binaryMetrics.precisionByThreshold().toDF("threshold", "precision")
  }

  /**
    * Returns a dataframe with two fields (threshold, recall) curve.
    * Every possible probability obtained in transforming the dataset are used
    * as thresholds used in calculating the recall.
    *
    * Note: This ignores instance weights (setting all to 1.0) from `FM.weightCol`.
    */
  @transient lazy val recallByThreshold: DataFrame = {
    binaryMetrics.recallByThreshold().toDF("threshold", "recall")
  }
}
/**
  * FactorAggregator computes the gradient and loss for binary FM loss function, as used
  * in binary classification for instances in sparse or dense vector in an online fashion.
  *
  * Two FactorAggregator can be merged together to have a summary of loss and gradient of
  * the corresponding joint dataset.
  *
  * @param bcCoefficients The broadcast coefficients corresponding to the features.
  * @param numFeatures number of features
  * @param fitIntercept Whether to fit an intercept term.
  */
private class FactorAggregator(bcCoefficients: Broadcast[Vector],
                               private val numFeatures: Int,
                               fitIntercept: Boolean,
                               isSparse: Boolean = false) extends Serializable with Logging {

  private val numFactors: Int = (bcCoefficients.value.size  - {if (fitIntercept) 1 else 0}) / numFeatures - 1
  private var weightSum = 0.0
  private var lossSum = 0.0
  private var iFactor = 0

  private val gradientSumArray =
    Array.ofDim[Double](numFeatures * (numFactors + 1) + (if (fitIntercept) 1 else 0))

  /**
    * Add a new training instance to this LogisticAggregator, and update the loss and gradient
    * of the objective function.
    *
    * @param instance The instance of data point to be added.
    * @return This LogisticAggregator object.
    */
  def add( instance: Instance): this.type = {
    instance match { case Instance(label, weight, features) =>
      require(numFeatures == features.size, s"Dimensions mismatch when adding new instance." +
        s" Expecting $numFeatures but got ${features.size}.")
      require(weight >= 0.0, s"instance weight, $weight has to be >= 0.0")

      if (weight == 0.0) return this

      val coefficients = bcCoefficients.value

      val coefficientsArray = coefficients match {
        case dv: DenseVector => dv.values
        case _ =>
          throw new IllegalArgumentException(
            s"coefficients only supports dense vector but got type ${coefficients.getClass}.")
      }
      val localGradientSumArray = gradientSumArray
      val factorSumArray = Array.ofDim[Double](numFactors)

      val margin = - {
        var sum = 0.0
        features.foreachActive { (index, value) =>
          if (value != 0.0) {
            sum += coefficientsArray(index) * value
            if (!isSparse || coefficientsArray(index) != 0.0) {
              iFactor = 0
              while (iFactor < numFactors) {
                val factorIndex = (iFactor + 1) * numFeatures + index
                sum -= 0.5 * coefficientsArray(factorIndex) * coefficientsArray(factorIndex) * value * value
                factorSumArray(iFactor) += coefficientsArray(factorIndex) * value
                iFactor += 1
              }
            }
          }
        }
        sum += 0.5 * factorSumArray.map(f => f * f).sum
        sum + {
          if (fitIntercept) coefficientsArray.last else 0.0
        }
      }

      val multiplier = weight * (1.0 / (1.0 + math.exp(margin)) - label)

      features.foreachActive { (index, value) =>
        if (value != 0.0) {
          localGradientSumArray(index) += multiplier * value
          if (!isSparse || coefficientsArray(index) != 0.0) {
            iFactor = 0
            while (iFactor < numFactors) {
              val factorIndex = (iFactor + 1) * numFeatures + index
              localGradientSumArray(factorIndex) += multiplier * value *
                (factorSumArray(iFactor) - coefficientsArray(factorIndex) * value)
              iFactor += 1
            }
          }
        }
      }

      if (fitIntercept) {
        localGradientSumArray(numFeatures * (numFactors + 1)) += multiplier
      }

      if (label > 0) {
        // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
        lossSum += weight * MLUtils.log1pExp(margin)
      } else {
        lossSum += weight * (MLUtils.log1pExp(margin) - margin)
      }
      weightSum += weight
    }
    this
  }

  /**
    * Merge another LogisticAggregator, and update the loss and gradient
    * of the objective function.
    * (Note that it's in place merging; as a result, `this` object will be modified.)
    *
    * @param other The other LogisticAggregator to be merged.
    * @return This LogisticAggregator object.
    */
  def merge(other: FactorAggregator): this.type = {
    require(numFeatures == other.numFeatures && numFactors == other.numFactors,
      s"Dimensions mismatch when merging with another FactorAggregator" +
      s"Expecting ($numFeatures, $numFactors) but got (${other.numFeatures}, ${other.numFactors}).")

    if (other.weightSum != 0.0) {
      weightSum += other.weightSum
      lossSum += other.lossSum

      var i = 0
      val localThisGradientSumArray = this.gradientSumArray
      val localOtherGradientSumArray = other.gradientSumArray
      val len = localThisGradientSumArray.length
      while (i < len) {
        localThisGradientSumArray(i) += localOtherGradientSumArray(i)
        i += 1
      }
    }
    this
  }

  def loss: Double = {
    require(weightSum > 0.0, s"The effective number of instances should be " +
      s"greater than 0.0, but $weightSum.")
    lossSum / weightSum
  }

  def gradient: Vector = {
    require(weightSum > 0.0, s"The effective number of instances should be " +
      s"greater than 0.0, but $weightSum.")
    val result = Vectors.dense(gradientSumArray.clone())
    scal(1.0 / weightSum, result)
    result
  }
}

/**
  * FactorCostFun implements Breeze's StochasticDiffFunction[T] for a FM function.
  * It returns the loss and gradient with L2 regularization at a particular point (coefficients).
  * It's used in Breeze's convex optimization routines.
  */
private class FactorCostFun(instances: RDD[Instance],
                            numFeatures: Int,
                            numFactors: Int,
                            fitIntercept: Boolean,
                            regParam1: Double,
                            regParam2: Double,
                            isSparse: Boolean = false) extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
    val coeffs = Vectors.fromBreeze(coefficients)

    val bcCoefficients = instances.context.broadcast(coeffs)
    val factorAggregator = {
      val seqOp = (c: FactorAggregator, instance: Instance) => c.add(instance)
      val combOp = (c1: FactorAggregator, c2: FactorAggregator) => c1.merge(c2)

      instances.treeAggregate(
        new FactorAggregator(bcCoefficients, numFeatures, fitIntercept, isSparse)
      )(seqOp, combOp)
    }

    val totalGradientArray = factorAggregator.gradient.toArray

    // regVal is the sum of coefficients squares excluding intercept for L2 regularization.
    val regVal = if (regParam1 == 0.0 && regParam2 == 0.0) {
      0.0
    } else {
      var sum = 0.0
      coeffs.foreachActive { (index, value) =>
        // If `fitIntercept` is true, the last term which is intercept doesn't
        // contribute to the regularization.
        if (index != numFeatures * (numFactors + 1)) {
          val regParam = if (index < numFeatures) regParam1 else regParam2
          // The following code will compute the loss of the regularization; also
          // the gradient of the regularization, and add back to totalGradientArray.
          sum += {
            totalGradientArray(index) += regParam * value
            0.5 * regParam * value * value
          }
        }
      }
      sum
    }

    // TODO log loss and regval
    println(s"FM loss: ${factorAggregator.loss}, regval: $regVal")
    (factorAggregator.loss + regVal, new BDV(totalGradientArray))
  }
}

object FMSparse {

  def makeSparse(w: BDV[Double], numFeatures: Int, numFactors: Int): Array[Double] = {
    require(w.length >= numFeatures * (numFactors + 1),
      "The length of weight vector must be equal with number of columns of factor matrix.")
    var i = 0
    while (i < numFeatures) {
      var j = 0
      if (w(i) == 0.0) {
        while (j < numFactors) {
          val index = (j + 1) * numFeatures + i
          w(index) = 0.0
          j += 1
        }
      }
      i += 1
    }
    w.toArray
  }
}
