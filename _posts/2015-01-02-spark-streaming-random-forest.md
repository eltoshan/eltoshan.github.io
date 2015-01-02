---
layout: post
title: "Implementing a Real-Time Predictive Alert System In Apache Spark"
date: 2015-01-02
---

With the release of [Spark 1.2.0](https://spark.apache.org/), MLlib now officially support Random Forests. Combined with Spark Streaming, we can now build high quality real-time predictive alert systems. In this post I will go through the process of preparing the data, building the model, and finally implementing the real-time system in Spark Streaming in Scala.

### Data Preparation

The dataset comes from the [2010 National Hospital Discharge Survey](http://www.cdc.gov/nchs/nhds/nhds_questionnaires.htm). The data includes various fields such as age, gender, procedured performed, etc. We will be using only the fields that are available prior to the patients discharge (so no discharge diagnoses), in order to predict the discharge status.

{% highlight scala %}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

// read file into RDD, split the string
val sc = new SparkContext()
val rawNHDSData = sc.textFile("NHDS10.PU.csv")
                    .map(_.split(",", -1)).filter(x => x(2) == "1")
{% endhighlight %}

We will have to convert the categorical features into an indexed list starting from $0$ to be able to use them in the Random Forest model.

{% highlight scala %}
// maps to reindex features to start from 0
val rawCatVars = Array(4, 5, 42, 43, 44) ++ (31 until 39)    
val rawCatMaps = rawCatVars.map { v =>
  rawNHDSData.map(x => x(v))
    .distinct().zipWithIndex.collect().toMap
}

// also prepare a map of the categorical feature arity
val rawCatArity = rawCatVars.map { v =>
  rawNHDSData.map(x => x(v)).distinct().count().toInt
}
val catVars = (4 until 17).toArray
val catMap = catVars.zip(rawCatArity).toMap

// save map of reindexed features to disk
import java.io.FileOutputStream
import java.io.ObjectOutputStream
val fosMap = new FileOutputStream("mapRF.obj")
val oosMap = new ObjectOutputStream(fosMap)
oosMap.writeObject(rawCatMaps)
oosMap.close
{% endhighlight %}

Now that we have the mapping for the categorical features, we can finally parse the raw data into an RDD of *LabeledPoint*.

{% highlight scala %}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._

// function to parse record and return a LabeledPoint
def parse(pieces: Array[Struing]) = {
  val ageUnit = pieces(2).toDouble
  val age = pieces(3).toDouble
  val dchgMonth = pieces(7).toDouble
  val daysIP = pieces(9).toDouble

  val reindexed = Array.fill[Double](rawCatVars.length)(0.0)
  for ( (v, i) <- rawCatVars.zipWithIndex ) {
    reindexed(i) = rawCatMaps(i)(pieces(v)).toDouble
  }

  val features = Array[Double](ageUnit, age,
    dchgMonth, daysIP) ++ reindexed

  val featureVector = Vectors.dense(features)
  val label = if (pieces(8) == "6") 1.0 else 0.0 // 1 for death, 0 otherwise

  LabeledPoint(label, featureVector)
}

// finally parse data to LabeledPoints
val NHDSData = rawNHDSData.map(parse(_))
{% endhighlight %}

The last step of the data preparation process is to split the data into training, validation, and test sets. The training set will be created using importance sampling in order to account for the class imbalance since there are much fewer deaths than other discharge statuses. We're also going to reserve $10\%$ of the data for our streaming system. 

{% highlight scala %}
// split to train/validation/test/streaming
val trainValTestStream = NHDSData.randomSplit(Array(0.7, 0.1, 0.1, 0.1),
  seed = 123123)

val trainSet = trainValTestStream(0)
val valData = trainValTestStream(1).cache()
val testData = trainValTestStream(2).cache()
val streamData = trainValTestStream(3)

// importance sampling on trainSet
val negFreq = trainSet.filter(x => x.label == 1.0).count().toDouble /
              trainSet.filter(x => x.label == 0.0).count().toDouble
val negData = trainSet.filter(x => x.label == 0.0)
                      .sample(withReplacement = false, fraction = negFreq)
val trainData = trainSet.filter(x => x.label == 1.0)
                        .union(negData)
                        .cache()

// save streaming portion to disk
streamData.map(l => (l.label +: l.features.toArray).mkString(","))
          .coalesce(1, true)
          .saveAsTextFile("streamData")
{% endhighlight %}

### Training Random Forest

With the data prepared into the form we need, now we can build and train the random forest model. We're more concerned with achieving a high true positive rate since death is a very, very bad thing. This will come at the cost of lower accuracy.

{% highlight scala %}
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.evaluation._

val numClasses = 2
val categoricalFeaturesInfo = catMap
val numTrees = 100
val featureSubsetStrategy = "auto"
val impurity = "gini"
val depth = 30
val bins = rawCatArity.max + 1

val modelRF = RandomForest.trainClassifier(
  trainData, numClasses, categoricalFeaturesInfo, numTrees,
  featureSubsetStrategy, impurity, depth, bins)
val predictionsAndLabelsRF = valData.map(r =>
  (modelRF.predict(r.features), r.label))
val metricsRF = new MulticlassMetrics(predictionsAndLabelsRF)

println((numTrees, impurity, depth, bins),
        metricsRF.truePositiveRate(1.0), metricsRF.precision)
println(metricsRF.confusionMatrix)

// save model to disk
val fosModel = new FileOutputStream("modelRF.obj")
val oosModel = new ObjectOutputStream(fosModel)
oosModel.writeObject(modelRF)
oosModel.close
{% endhighlight %}

The confusion matrix shows what we more or less wanted: high true-positive rate with the cost of increasing false-positive rate. The rows are actual labels of $0/1$, and columns are predictive labels.

{% highlight sh %}
5369.0  7859.0                                                                 
17.0    258.0  
{% endhighlight %}

### Real Time Predictions

Now that we have the model, it's time to create the real time streaming system. Most Unix based operating systems come with a convenient utility called netcat that we will use to open a socket that publishes the streaming data we had previously reserved with a simple shell script.

{% highlight sh %}
#!/bin/bash
while read LINE; do echo $LINE; sleep 10; done < streamData/part-00000 | nc -lk 9999
{% endhighlight %}

First thing we need to do is create a streaming context, listen on the port that netcat is publishing to, and read in the model we had saved.

{% highlight scala %}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._

import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model.RandomForestModel

import java.io.FileInputStream
import java.io.ObjectInputStream

val sc = new SparkContext()
val ssc = new StreamingContext(sc, Seconds(10))

// load model
val oisModel = new ObjectInputStream(new FileInputStream("modelRF.obj"))
val modelRF = oisModel.readObject().asInstanceOf[RandomForestModel]

// load feature maps
val oisMap = new ObjectInputStream(new FileInputStream("mapRF.obj"))
val mapRF = oisMap.readObject().asInstanceOf[Array[Map[String, Int]]]
                  .map(_.map(_.swap))

// create DStreams
val events = ssc.socketTextStream("localhost", 9999)
val lines = events.map(_.split(",", -1).map(_.toDouble))
                  .map(x => LabeledPoint(x(0), Vectors.dense(x.tail)))
{% endhighlight %}

Last thing to do before we compile and run the streaming system is actually taking the features and predicting the discharge status using the Random Forest model.

{% highlight scala %}
// print raw features and prediction/label
def getRawVal(pieces: Array[Double]) = {
  val origFeatures = Array.fill[String](mapRF.length)(" ")
  for ( i <- 0 until mapRF.length ) {
    origFeatures(i) = mapRF(i)(pieces(i+4).toInt)
  }
  val features = pieces.slice(0, 4).map(_.toString) ++ origFeatures
  features.mkString(",")
}
val features = lines.map(r => getRawVal(r.features.toArray))
features.print()

val predictionsAndLabels = lines.map(r =>
  (modelRF.predict(r.features), r.label))
predictionsAndLabels.print()

ssc.start()
ssc.awaitTermination()
{% endhighlight %}

Finally we'll see output like below. One line of input features followed by the predicted status and actual status.

{% highlight bash %}
{% raw %}
-------------------------------------------
Time: 1419894950000 ms
-------------------------------------------
1.0,74.0,5.0,2.0,2,1,1,6,5990-,,,,,,,,

-------------------------------------------
Time: 1419894950000 ms
-------------------------------------------
(0.0,0.0)

14/12/29 18:16:00 WARN BlockManager: Block input-0-1419894960200 replicated to only 0 peer(s) instead of 1 peers
-------------------------------------------
Time: 1419894960000 ms
-------------------------------------------
1.0,86.0,7.0,1.0,2,1,1,6,4275-,9671,9604,9917,9918,,,,

-------------------------------------------
Time: 1419894960000 ms
-------------------------------------------
(1.0,1.0)
{% endraw %}
{% endhighlight %}