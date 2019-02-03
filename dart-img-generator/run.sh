#!/bin/bash
mvn clean install
cp target/dart-img-generator.jar .
nohup java -jar -Dlogging.file=/home/salgmachine/dart-net/dart-net.log dart-img-generator-0.0.1-SNAPSHOT.jar
