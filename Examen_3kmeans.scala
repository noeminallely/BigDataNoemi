//1-importar sesion spark
import org.apache.spark.sql.SparkSession
//2-recuperar los errores
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
// 3-instancias de la sesion de spark
val spark = SparkSession.builder().getOrCreate()
//4- importar libreria k-means
import org.apache.spark.ml.clustering.KMeans
//5-cargar data set
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")
dataset.show()
// 6. seleccionar las columnas para el entrenamiento
val feature_data = dataset.select($"Fresh",$"Milk",$"Grocery",$"Frozen",$"Detergents_Paper",$"Delicassen")
//7. importar vectorassambler
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
//8.crear un objeto vectorassambler
val VectorAssembler= new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")
//9. utilizar el objeto assembler para transformar feature_data
val training = VectorAssembler.transform(feature_data).select("features")
//10. crear un modelo Kmeans con K=3
val kmeans = new KMeans().setK(3).setSeed(1L)
// 11.  Evaluar los grupos utilizando WSSE
val model = kmeans.fit(training)
val WSSSE = model.computeCost(training)
//12. Mostrar los resultados
println(s"The within set sum of squared errors was: ${WSSSE} ")
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
