#!/bin/bash
cd /home/paperspace/dart-net
git pull
## mvn clean install -DskipTests=true -f /home/paperspace/dart-net/dart-img-generator/pom.xml
mvn clean install -DskipTests=true -Pcuda -f /home/paperspace/dart-net/dart-net-demo-1/pom.xml
/home/paperspace/jdk8/bin/java -jar -Ddartnet.input=/home/paperspace/dart-net/dart-net
