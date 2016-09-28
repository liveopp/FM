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
import org.apache.spark.annotation.{Experimental, Since}
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
  * Params for logistic regression.
  */
private[classification] trait FactorizationMachineParams extends ProbabilisticClassifierParams
  with HasRegParam with HasElasticNetParam with HasMaxIter with HasFitIntercept with HasTol
  with HasStandardization with HasWeightCol with HasThreshold {

  /**
    * Set threshold in binary classification, in range [0, 1].
    *
    * If the estimated probability of class label 1 is > threshold, then predict 1, else 0.
    * A high threshold encourages the model to predict 0 more often;
    * a low threshold encourages the model to predict 1 more often.
    *
    * Note: Calling this with threshold p is equivalent to calling `setThresholds(Array(1-p, p))`.
    *       When [[setThreshold()]] is called, any user-set value for [[thresholds]] will be cleared.
    *       If both [[threshold]] and [[thresholds]] are set in a ParamMap, then they must be
    *       equivalent.
    *
    * Default is 0.5.
    * @group setParam
    */
  def setThreshold(value: Double): this.type = {
    if (isSet(thresholds)) clear(thresholds)
    set(threshold, value)
  }

  /**
    * Get threshold for binary classification.
    *
    * If [[thresholds]] is set with length 2 (i.e., binary classification),
    * this returns the equivalent threshold: {{{1 / (1 + thresholds(0) / thresholds(1))}}}.
    * Otherwise, returns [[threshold]] if set, or its default value if unset.
    *
    * @group getParam
    * @throws IllegalArgumentException if [[thresholds]] is set to an array of length other than 2.
    */
  override def getThreshold: Double = {
    checkThresholdConsistency()
    if (isSet(thresholds)) {
      val ts = $(thresholds)
      require(ts.length == 2, "Logistic Regression getThreshold only applies to" +
        " binary classification, but thresholds has length != 2.  thresholds: " + ts.mkString(","))
      1.0 / (1.0 + ts(0) / ts(1))
    } else {
      $(threshold)
    }
  }

  /**
    * Set thresholds in multiclass (or binary) classification to adjust the probability of
    * predicting each class. Array must have length equal to the number of classes, with values >= 0.
    * The class with largest value p/t is predicted, where p is the original probability of that
    * class and t is the class' threshold.
    *
    * Note: When [[setThresholds()]] is called, any user-set value for [[threshold]] will be cleared.
    *       If both [[threshold]] and [[thresholds]] are set in a ParamMap, then they must be
    *       equivalent.
    *
    * @group setParam
    */
  def setThresholds(value: Array[Double]): this.type = {
    if (isSet(threshold)) clear(threshold)
    set(thresholds, value)
  }

  /**
    * Get thresholds for binary or multiclass classification.
    *
    * If [[thresholds]] is set, return its value.
    * Otherwise, if [[threshold]] is set, return the equivalent thresholds for binary
    * classification: (1-threshold, threshold).
    * If neither are set, throw an exception.
    *
    * @group getParam
    */
  override def getThresholds: Array[Double] = {
    checkThresholdConsistency()
    if (!isSet(thresholds) && isSet(threshold)) {
      val t = $(threshold)
      Array(1-t, t)
    } else {
      $(thresholds)
    }
  }

  /**
    * If [[threshold]] and [[thresholds]] are both set, ensures they are consistent.
    * @throws IllegalArgumentException if [[threshold]] and [[thresholds]] are not equivalent
    */
  protected def checkThresholdConsistency(): Unit = {
    if (isSet(threshold) && isSet(thresholds)) {
      val ts = $(thresholds)
      require(ts.length == 2, "Logistic Regression found inconsistent values for threshold and" +
        s" thresholds.  Param threshold is set (${$(threshold)}), indicating binary" +
        s" classification, but Param thresholds is set with length ${ts.length}." +
        " Clear one Param value to fix this problem.")
      val t = 1.0 / (1.0 + ts(0) / ts(1))
      require(math.abs($(threshold) - t) < 1E-5, "Logistic Regression getThreshold found" +
        s" inconsistent values for threshold (${$(threshold)}) and thresholds (equivalent to $t)")
    }
  }

  override def validateParams(): Unit = {
    checkThresholdConsistency()
  }
}

/**
  * Factorization machines.
  * Currently, this class only supports binary classification.  It will support multiclass
  * in the future.
  */
class FactorizationMachine (override val uid: String, val numFactors: Int)
  extends ProbabilisticClassifier[Vector, FactorizationMachine, FactorizationMachineModel]
    with FactorizationMachineParams with DefaultParamsWritable with Logging {

  require(numFactors > 0, "Factor number must > 0")

  def this(numFactors: Int) = this(Identifiable.randomUID("fm"), numFactors)

  def this() = this(20)

  /**
    * Set the regularization parameter.
    * Default is 0.0.
    * @group setParam
    */
  def setRegParam(value: Double): this.type = set(regParam, value)
  setDefault(regParam -> 0.0)

  /**
    * Set the ElasticNet mixing parameter.
    * For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.
    * For 0 < alpha < 1, the penalty is a combination of L1 and L2.
    * Default is 0.0 which is an L2 penalty.
    * @group setParam
    */
  @Since("1.4.0")
  def setElasticNetParam(value: Double): this.type = set(elasticNetParam, value)
  setDefault(elasticNetParam -> 0.0)

  /**
    * Set the maximum number of iterations.
    * Default is 100.
    * @group setParam
    */
  @Since("1.2.0")
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  /**
    * Set the convergence tolerance of iterations.
    * Smaller value will lead to higher accuracy with the cost of more iterations.
    * Default is 1E-6.
    * @group setParam
    */
  @Since("1.4.0")
  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  /**
    * Whether to fit an intercept term.
    * Default is true.
    * @group setParam
    */
  @Since("1.4.0")
  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)
  setDefault(fitIntercept -> true)

  /**
    * Whether to standardize the training features before fitting the model.
    * The coefficients of models will be always returned on the original scale,
    * so it will be transparent for users. Note that with/without standardization,
    * the models should be always converged to the same solution when no regularization
    * is applied. In R's GLMNET package, the default behavior is true as well.
    * Default is true.
    * @group setParam
    */
  @Since("1.5.0")
  def setStandardization(value: Boolean): this.type = set(standardization, value)
  setDefault(standardization -> true)

  @Since("1.5.0")
  override def setThreshold(value: Double): this.type = super.setThreshold(value)

  @Since("1.5.0")
  override def getThreshold: Double = super.getThreshold

  /**
    * Whether to over-/under-sample training instances according to the given weights in weightCol.
    * If not set or empty String, all instances are treated equally (weight 1.0).
    * Default is not set, so all instances have weight one.
    * @group setParam
    */
  @Since("1.6.0")
  def setWeightCol(value: String): this.type = set(weightCol, value)

  @Since("1.5.0")
  override def setThresholds(value: Array[Double]): this.type = super.setThresholds(value)

  @Since("1.5.0")
  override def getThresholds: Array[Double] = super.getThresholds

  override protected[spark] def train(dataset: Dataset[_]): FactorizationMachineModel = {
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    train(dataset, handlePersistence)
  }

  protected[spark] def train(dataset: Dataset[_], handlePersistence: Boolean):
  FactorizationMachineModel = {
    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val instances: RDD[Instance] =
      dataset.select(col($(labelCol)).cast(DoubleType), w, col($(featuresCol))).rdd.map {
        case Row(label: Double, weight: Double, features: Vector) =>
          Instance(label, weight, features)
      }

    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val instr = Instrumentation.create(this, instances)
    instr.logParams(regParam, elasticNetParam, standardization, threshold,
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

    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    instr.logNumClasses(numClasses)
    instr.logNumFeatures(numFeatures)

    val (coefficients, intercept, objectiveHistory) = {
      if (numInvalid != 0) {
        val msg = s"Classification labels should be in {0 to ${numClasses - 1} " +
          s"Found $numInvalid invalid labels."
        logError(msg)
        throw new SparkException(msg)
      }

      if (numClasses > 2) {
        val msg = s"Currently, LogisticRegression with ElasticNet in ML package only supports " +
          s"binary classification. Found $numClasses in the input dataset."
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

        val regParamL1 = $(elasticNetParam) * $(regParam)
        val regParamL2 = (1.0 - $(elasticNetParam)) * $(regParam)

        val costFun = new FactorCostFun(instances, numFeatures, numFactors,
          numClasses, $(fitIntercept), regParamL2)

        val optimizer = if ($(elasticNetParam) == 0.0 || $(regParam) == 0.0) {
          new BreezeLBFGS[BDV[Double]]($(maxIter), 10, $(tol))
        } else {
          throw new NotImplementedError("L1 regularation is not implemented")
        }

        val initialCoefficientsArray = Array.fill[Double](
          numFeatures * (numFactors + 1) + (if ($(fitIntercept)) 1 else 0))(
          1 / math.sqrt(numFactors) * (Random.nextDouble - 0.5))
        var i = 0
        while (i < numFeatures) {
          initialCoefficientsArray(i) = 0.0
          i += 1
        }
        val initialCoefficientsWithIntercept = Vectors.dense(initialCoefficientsArray)

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
          initialCoefficientsWithIntercept.toArray(numFeatures) = math.log(
            histogram(1) / histogram(0))
        }

        val states = optimizer.iterations(new CachedDiffFunction(costFun),
          initialCoefficientsWithIntercept.asBreeze.toDenseVector)

        /*
           Note that in Logistic Regression, the objective history (loss + regularization)
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

        if (!state.actuallyConverged) {
          logWarning("FactorizationMachine training finished but the result " +
            s"is not converged because: ${state.convergedReason.get.reason}")
        }

        /*
           The coefficients are trained in the scaled space; we're converting them back to
           the original space.
           Note that the intercept in scaled space and original space is the same;
           as a result, no scaling is needed.
         */
        val rawCoefficients = state.x.toArray.clone()

        if ($(fitIntercept)) {
          (Vectors.dense(rawCoefficients.dropRight(1)).compressed, rawCoefficients.last,
            arrayBuilder.result())
        } else {
          (Vectors.dense(rawCoefficients).compressed, 0.0, arrayBuilder.result())
        }
      }
    }

    if (handlePersistence) instances.unpersist()

    val model = copyValues(new FactorizationMachineModel(uid, numFactors, coefficients, intercept))
    val (summaryModel, probabilityColName) = model.findSummaryModelAndProbabilityCol()
    val fmSummary = new BinaryFactorizationMachineTrainingSummary(
      summaryModel.transform(dataset),
      probabilityColName,
      $(labelCol),
      $(featuresCol),
      objectiveHistory)
    val m = model.setSummary(fmSummary)
    instr.logSuccess(m)
    m
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): FactorizationMachine = defaultCopy(extra)
}

@Since("1.6.0")
object FactorizationMachine extends DefaultParamsReadable[FactorizationMachine] {

  @Since("1.6.0")
  override def load(path: String): FactorizationMachine = super.load(path)
}

/**
  * Model produced by [[FactorizationMachine]].
  */
class FactorizationMachineModel private[spark](
                                                override val uid: String,
                                                val numFactors: Int,
                                                val coefficients: Vector,
                                                val intercept: Double)
  extends ProbabilisticClassificationModel[Vector, FactorizationMachineModel]
    with FactorizationMachineParams with MLWritable {

  override def setThreshold(value: Double): this.type = super.setThreshold(value)

  override def getThreshold: Double = super.getThreshold

  override def setThresholds(value: Array[Double]): this.type = super.setThresholds(value)

  override def getThresholds: Array[Double] = super.getThresholds

  override val numFeatures: Int = coefficients.size / (numFactors + 1)

  private val factorNorms = {
    val coeffs = BDV(coefficients.toArray)
    val factorMatrix = coeffs(numFeatures until numFeatures * (numFactors + 1))
      .toDenseMatrix
      .reshape(numFeatures, numFactors)
    val squreMatrix = factorMatrix :* factorMatrix
    Vectors.fromBreeze(sum(squreMatrix(*, ::)))
  }

  /** Margin (rawPrediction) for class label 1.  For binary classification only. */
  private val margin: Vector => Double = (features) => {
    val factorSumArray = Array.ofDim[Double](numFactors)
    val coefficientsArray = coefficients.toArray
    val numFeatures = features.size
    var sum = 0.0
    features.foreachActive { (index, value) =>
      if (value != 0.0) {
        sum += coefficientsArray(index) * value
        sum -= 0.5 * factorNorms(index) * value * value
        var iFactor = 0
        while (iFactor < numFactors) {
          val factorIndex = (iFactor + 1) * numFeatures + index
          factorSumArray(iFactor) += coefficientsArray(factorIndex) * value
          iFactor += 1
        }
      }
    }
    sum += 0.5 * factorSumArray.map(f => f * f).sum
    sum + intercept
  }

  /** Score (probability) for class label 1.  For binary classification only. */
  private val score: Vector => Double = (features) => {
    val m = margin(features)
    1.0 / (1.0 + math.exp(-m))
  }


  override val numClasses: Int = 2

  private var trainingSummary: Option[FactorizationMachineTrainingSummary] = None

  /**
    * Gets summary of model on training set. An exception is
    * thrown if `trainingSummary == None`.
    */
  @Since("1.5.0")
  def summary: FactorizationMachineTrainingSummary = trainingSummary.getOrElse {
    throw new SparkException("No training summary available for this LogisticRegressionModel")
  }

  /**
    * If the probability column is set returns the current model and probability column,
    * otherwise generates a new column and sets it as the probability column on a new copy
    * of the current model.
    */
  private[classification] def findSummaryModelAndProbabilityCol():
  (FactorizationMachineModel, String) = {
    $(probabilityCol) match {
      case "" =>
        val probabilityColName = "probability_" + java.util.UUID.randomUUID.toString
        (copy(ParamMap.empty).setProbabilityCol(probabilityColName), probabilityColName)
      case p => (this, p)
    }
  }

  private[classification] def setSummary(
                                          summary: FactorizationMachineTrainingSummary): this.type = {
    this.trainingSummary = Some(summary)
    this
  }

  /** Indicates whether a training summary exists for this model instance. */
  @Since("1.5.0")
  def hasSummary: Boolean = trainingSummary.isDefined

  /**
    * Evaluates the model on a test dataset.
    * @param dataset Test dataset to evaluate model on.
    */
  @Since("2.0.0")
  def evaluate(dataset: Dataset[_]): FactorizationMachineSummary = {
    // Handle possible missing or invalid prediction columns
    val (summaryModel, probabilityColName) = findSummaryModelAndProbabilityCol()
    new BinaryFactorizationMachineSummary(summaryModel.transform(dataset),
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

  @Since("1.4.0")
  override def copy(extra: ParamMap): FactorizationMachineModel = {
    val newModel = copyValues(new FactorizationMachineModel(uid, numFactors, coefficients, intercept), extra)
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
    * For [[FactorizationMachineModel]], this does NOT currently save the training [[summary]].
    * An option to save [[summary]] may be added in the future.
    *
    * This also does not save the [[parent]] currently.
    */
  @Since("1.6.0")
  override def write: MLWriter = new FactorizationMachineModel.FactorizationMachineModelWriter(this)
}


@Since("1.6.0")
object FactorizationMachineModel extends MLReadable[FactorizationMachineModel] {

  @Since("1.6.0")
  override def read: MLReader[FactorizationMachineModel] = new LogisticRegressionModelReader

  @Since("1.6.0")
  override def load(path: String): FactorizationMachineModel = super.load(path)

  /** [[MLWriter]] instance for [[FactorizationMachineModel]] */
  private[FactorizationMachineModel]
  class FactorizationMachineModelWriter(instance: FactorizationMachineModel)
    extends MLWriter with Logging {

    private case class Data(
                             numClasses: Int,
                             numFeatures: Int,
                             intercept: Double,
                             coefficients: Vector)

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      // Save model data: numClasses, numFeatures, intercept, coefficients
      val data = Data(instance.numClasses, instance.numFeatures, instance.intercept,
        instance.coefficients)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class LogisticRegressionModelReader
    extends MLReader[FactorizationMachineModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[FactorizationMachineModel].getName

    override def load(path: String): FactorizationMachineModel = {
      throw new SparkException("model loder not implemented")
    }
  }
}

/**
  * Abstraction for multinomial Logistic Regression Training results.
  * Currently, the training summary ignores the training weights except
  * for the objective trace.
  */
sealed trait FactorizationMachineTrainingSummary extends FactorizationMachineSummary {

  /** objective function (scaled loss + regularization) at each iteration. */
  def objectiveHistory: Array[Double]

  /** Number of training iterations until termination */
  def totalIterations: Int = objectiveHistory.length

}

/**
  * Abstraction for Logistic Regression Results for a given model.
  */
sealed trait FactorizationMachineSummary extends Serializable {

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
@Experimental
@Since("1.5.0")
class BinaryFactorizationMachineTrainingSummary private[classification](
                                                                        predictions: DataFrame,
                                                                        probabilityCol: String,
                                                                        labelCol: String,
                                                                        featuresCol: String,
                                                                        @Since("1.5.0") val objectiveHistory: Array[Double])
  extends BinaryFactorizationMachineSummary(predictions, probabilityCol, labelCol, featuresCol)
    with FactorizationMachineTrainingSummary {

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
@Since("1.5.0")
class BinaryFactorizationMachineSummary private[classification](
                                                                @Since("1.5.0") @transient override val predictions: DataFrame,
                                                                @Since("1.5.0") override val probabilityCol: String,
                                                                @Since("1.5.0") override val labelCol: String,
                                                                @Since("1.6.0") override val featuresCol: String) extends FactorizationMachineSummary {


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
    * Note: This ignores instance weights (setting all to 1.0) from `LogisticRegression.weightCol`.
    *       This will change in later Spark versions.
    */
  @Since("1.5.0")
  @transient lazy val roc: DataFrame = binaryMetrics.roc().toDF("FPR", "TPR")

  /**
    * Computes the area under the receiver operating characteristic (ROC) curve.
    *
    * Note: This ignores instance weights (setting all to 1.0) from `LogisticRegression.weightCol`.
    *       This will change in later Spark versions.
    */
  @Since("1.5.0")
  lazy val areaUnderROC: Double = binaryMetrics.areaUnderROC()

  /**
    * Returns the precision-recall curve, which is a Dataframe containing
    * two fields recall, precision with (0.0, 1.0) prepended to it.
    *
    * Note: This ignores instance weights (setting all to 1.0) from `LogisticRegression.weightCol`.
    *       This will change in later Spark versions.
    */
  @Since("1.5.0")
  @transient lazy val pr: DataFrame = binaryMetrics.pr().toDF("recall", "precision")

  /**
    * Returns a dataframe with two fields (threshold, F-Measure) curve with beta = 1.0.
    *
    * Note: This ignores instance weights (setting all to 1.0) from `LogisticRegression.weightCol`.
    *       This will change in later Spark versions.
    */
  @Since("1.5.0")
  @transient lazy val fMeasureByThreshold: DataFrame = {
    binaryMetrics.fMeasureByThreshold().toDF("threshold", "F-Measure")
  }

  /**
    * Returns a dataframe with two fields (threshold, precision) curve.
    * Every possible probability obtained in transforming the dataset are used
    * as thresholds used in calculating the precision.
    *
    * Note: This ignores instance weights (setting all to 1.0) from `LogisticRegression.weightCol`.
    *       This will change in later Spark versions.
    */
  @Since("1.5.0")
  @transient lazy val precisionByThreshold: DataFrame = {
    binaryMetrics.precisionByThreshold().toDF("threshold", "precision")
  }

  /**
    * Returns a dataframe with two fields (threshold, recall) curve.
    * Every possible probability obtained in transforming the dataset are used
    * as thresholds used in calculating the recall.
    *
    * Note: This ignores instance weights (setting all to 1.0) from `LogisticRegression.weightCol`.
    *       This will change in later Spark versions.
    */
  @Since("1.5.0")
  @transient lazy val recallByThreshold: DataFrame = {
    binaryMetrics.recallByThreshold().toDF("threshold", "recall")
  }
}
/**
  * FactorAggregator computes the gradient and loss for binary logistic loss function, as used
  * in binary classification for instances in sparse or dense vector in an online fashion.
  *
  * Note that multinomial logistic loss is not supported yet!
  *
  * Two LogisticAggregator can be merged together to have a summary of loss and gradient of
  * the corresponding joint dataset.
  *
  * @param numClasses the number of possible outcomes for k classes classification problem in
  *                   Multinomial Logistic Regression.
  * @param fitIntercept Whether to fit an intercept term.
  */
private class FactorAggregator(
                                private val numFeatures: Int,
                                private val numFactors: Int,
                                numClasses: Int,
                                fitIntercept: Boolean) extends Serializable with Logging {

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
    * @param coefficients The coefficients corresponding to the features.
    * @return This LogisticAggregator object.
    */
  def add(
           instance: Instance,
           coefficients: Vector,
           factorNorms: Vector): this.type = {
    instance match { case Instance(label, weight, features) =>
      require(numFeatures == features.size, s"Dimensions mismatch when adding new instance." +
        s" Expecting $numFeatures but got ${features.size}.")
      require(weight >= 0.0, s"instance weight, $weight has to be >= 0.0")

      if (weight == 0.0) return this

      val coefficientsArray = coefficients match {
        case dv: DenseVector => dv.values
        case _ =>
          throw new IllegalArgumentException(
            s"coefficients only supports dense vector but got type ${coefficients.getClass}.")
      }
      val localGradientSumArray = gradientSumArray
      val factorSumArray = Array.ofDim[Double](numFactors)

      numClasses match {
        case 2 =>
          // For Binary Logistic Regression.
          val margin = - {
            var sum = 0.0
            features.foreachActive { (index, value) =>
              if (value != 0.0) {
                sum += coefficientsArray(index) * value
                sum -= 0.5 * factorNorms(index) * value * value
                iFactor = 0
                while (iFactor < numFactors) {
                  val factorIndex = (iFactor + 1) * numFeatures + index
                  factorSumArray(iFactor) += coefficientsArray(factorIndex) * value
                  iFactor += 1
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
              iFactor = 0
              while (iFactor < numFactors) {
                val factorIndex = (iFactor + 1) * numFeatures + index
                localGradientSumArray(factorIndex) += multiplier * value *
                  (factorSumArray(iFactor) - coefficientsArray(factorIndex) * value)
                iFactor += 1
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
        case _ =>
          new NotImplementedError("LogisticRegression with ElasticNet in ML package " +
            "only supports binary classification for now.")
      }
      weightSum += weight
      this
    }
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
    require(numFeatures == other.numFeatures, s"Dimensions mismatch when merging with another " +
      s"LeastSquaresAggregator. Expecting $numFeatures but got ${other.numFeatures}.")

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
    println(s"weightSum : $weightSum, lossSum: $lossSum")
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
  * LogisticCostFun implements Breeze's DiffFunction[T] for a multinomial logistic loss function,
  * as used in multi-class classification (it is also used in binary logistic regression).
  * It returns the loss and gradient with L2 regularization at a particular point (coefficients).
  * It's used in Breeze's convex optimization routines.
  */
private class FactorCostFun(
                             instances: RDD[Instance],
                             numFeatures: Int,
                             numFactors: Int,
                             numClasses: Int,
                             fitIntercept: Boolean,
                             regParamL2: Double) extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
    val coeffs = Vectors.fromBreeze(coefficients)
    val factorMatrix = coefficients(numFeatures until numFeatures * (numFactors + 1))
      .toDenseMatrix
      .reshape(numFeatures, numFactors)
    val squreMatrix = factorMatrix :* factorMatrix
    val factorNorms = Vectors.fromBreeze(sum(squreMatrix(*, ::)))

    val factorAggregator = {
      val seqOp = (c: FactorAggregator, instance: Instance) =>
        c.add(instance, coeffs, factorNorms)
      val combOp = (c1: FactorAggregator, c2: FactorAggregator) => c1.merge(c2)

      instances.treeAggregate(
        new FactorAggregator(numFeatures, numFactors, numClasses, fitIntercept)
      )(seqOp, combOp)
    }

    val totalGradientArray = factorAggregator.gradient.toArray

    // regVal is the sum of coefficients squares excluding intercept for L2 regularization.
    val regVal = if (regParamL2 == 0.0) {
      0.0
    } else {
      var sum = 0.0
      coeffs.foreachActive { (index, value) =>
        // If `fitIntercept` is true, the last term which is intercept doesn't
        // contribute to the regularization.
        if (index != numFeatures * (numFactors + 1)) {
          // The following code will compute the loss of the regularization; also
          // the gradient of the regularization, and add back to totalGradientArray.
          sum += {
            totalGradientArray(index) += regParamL2 * value
            value * value
          }
        }
      }
      0.5 * regParamL2 * sum
    }

    println(s"factorAggregator loss: ${factorAggregator.loss}, regval: $regVal")
    (factorAggregator.loss + regVal, new BDV(totalGradientArray))
  }
}

