name := "LinearRegressionProject"

version := "0.1"

scalaVersion := "2.12.10"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.1.1",
  "org.apache.spark" %% "spark-sql" % "3.1.1",
  "org.apache.spark" %% "spark-mllib" % "3.1.1",
  "org.scalanlp" %% "breeze" % "1.2",
  "org.scalatest" %% "scalatest" % "3.2.9" % Test
)
