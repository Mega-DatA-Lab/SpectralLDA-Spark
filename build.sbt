name := "SpectralLDA-Tensor"

version := "1.2.6"

scalaVersion := "2.11.12"

libraryDependencies ++= Seq(
    "com.nativelibs4java" %% "scalaxy-loops" % "[0.3.4,)",
    "com.github.scopt" %% "scopt" % "[3.4.0,)",
    "org.scalatest" %% "scalatest" % "3.0.5" % "test"
)

val sparkVersion = "[2.3.0,)"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided"
)
