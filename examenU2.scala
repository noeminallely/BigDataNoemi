import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import spark.implicits._
import org.apache.spark.sql.Column
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.Pipeline
//

val spar = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","false").option("inferSchema","true")csv("iris.csv")
//
val df = spark.read.option("inferSchema","true").csv("iris.csv").toDF("SepalLength","SepalWidth","PetalLength","PetalWidth","Species"
)
///
val newcol = when($"Species".contains("Iris-setosa"), 1.0).otherwise(when($"Species".contains("Iris-virginica"), 3.0).otherwise(2.0))

val df2= df.select("SepalLength","SepalWidth","PetalLength","PetalWidth","Species")


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
val assembler = new VectorAssembler()  .setInputCols(Array(
  "SepalLength",
  "SepalWidth",
  "PetalLength",
  "PetalWidth",
  "etiqueta")).setOutputCol("features")
//Transformar datos
val features = assembler.transform(newdf)
features.show(5)


val header = df2.first


// Load the data stored in LIBSVM format as a DataFrame.


// Split the data into train and test
val splits = df2.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4
// and output of size 3 (classes)
val layers = Array[Int](4, 5, 4, 3)

// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// train the model
val model = trainer.fit(train)

// compute accuracy on the test set
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
