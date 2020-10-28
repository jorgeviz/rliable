# Spark installation for USC HPC DiscoveryServer 

# Load Java module
echo "module load jdk" >> ~/.bashrc

# copy and extract Spark files
cd ~/
mkdir spark/
cd spark/
wget https://apache.osuosl.org/spark/spark-3.0.1/spark-3.0.1-bin-hadoop2.7.tgz
tar -xvf spark-3.0.1-bin-hadoop2.7.tgz
echo "export SPARK_HOME=$HOME/spark/spark-3.0.1-bin-hadoop2.7" >> ~/.bashrc

# source bashrc
source ~/.bashrc

# validate version
$SPARK_HOME/bin/pyspark --version

