import org.apache.spark.sql.Encoders
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession

case class Initial(sepalLength: Option[Double], sepalWidth: Option[Double], petalLength: Option[Double], petalWidth: Option[Double], species: Option[String])
case class Final(sepalLength : Double, sepalWidth : Double, petalLength : Double, petalWidth : Double, species: Double)

//se crea el inicio de sesion
val conf = new SparkConf().setMaster("local[*]").setAppName("IrisSpark")
val sparkSession = SparkSession.builder.config(conf = conf).appName("spark session example").getOrCreate()
val path = "iris.csv"
//se coloca nombre al encabezado de la tabla para que sea mas facil su modificacion
var irisSchema2 = Encoders.product[Initial].schema
val iris: DataFrame = sparkSession.read.option("header","true").option("inferSchema", "true").schema(irisSchema2).csv(path)
iris.show()


// Un ensamblador convierte los valores de entrada a un vector
// Un vector es lo que el algoritmo ML lee para entrenar un modelo.
// Establece las columnas de entrada de las cuales se supone que debemos leer los valores
// Establece el nombre de la columna donde se almacenará el vector
val assembler = new VectorAssembler().setInputCols(Array("sepalLength", "sepalWidth", "petalLength", "petalWidth", "species")).setOutputCol("features")


/* Antes de que podamos llamar al ensamblador, tendremos que convertir todos
 * Los valores de cadena para doblar y eliminar cualquier valor nulo.
 * Esta función limpiará esos datos por nosotros.
 */

def autobot(in: Initial) = Final(
    in.sepalLength.map(_.toDouble).getOrElse(0),
    in.sepalWidth.map(_.toDouble).getOrElse(0),
    in.petalLength.map(_.toDouble).getOrElse(0),
    in.petalWidth.map(_.toDouble).getOrElse(0),
    in.species match {
      case Some("Iris-versicolor") => 1;
      case Some("Iris-virginica") => 2;
      case Some("Iris-setosa") => 3;
      case _ => 3;
    }
  )

  // Una vez que tengamos todas las funciones listas, este es el momento de
  // aplicarlos y limpiar los datos.
val data = assembler.transform(iris.as[Initial].map(autobot))
data.show()

/*hasta aqui termina la limpieza de datos */


/*Comienza el uso de MLP DE LA LIBRERIA SPARK  y los demas puntos*/

// Indice etiquetas, agregando metadatos a la columna de etiquetas.
// Ajustar en el conjunto de datos completo para incluir todas las etiquetas en el índice.
val labelIndexer = new StringIndexer().setInputCol("species").setOutputCol("indexedLabel").fit(data)

// Identificar automáticamente las características categóricas, e indexarlas.
// Establezca maxCategories para que las características con> 4 valores distintos se traten como continuas.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)


//El conjunto de datos se divide en una parte utilizada para entrenar el modelo (70%) y otra parte para las prueba (30%).
val Array(trainingData, testData) = data randomSplit Array(0.7, 0.3)

// Entrena un modelo de RandomForest.
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

// Convertir las etiquetas indexadas de nuevo a las etiquetas originales.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Cadena de indexadores y bosque en un Pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))


// Modelo de entrenamiento. Esto también ejecuta los indexadores.
val model = pipeline fit trainingData

//Crea predicciones
val predictions = model transform testData

// muestra las predictions seleccionadas
predictions.select("species", "prediction", "features").show()
predictions.show()

// Seleccionar (predicción, etiqueta verdadera) y calcular el error de prueba.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

// Medir la precisión del modelo utilizando el evaluador.
val accuracy = evaluator evaluate predictions

// Imprime la exactitud en la consola
println("Test Error = " + (accuracy))
