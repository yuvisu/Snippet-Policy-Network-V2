####################################
#   GET ICBEB 2018 (CPSC2018) DATABASE
####################################
mkdir -p tmp_data
cd tmp_data
wget http://2018.icbeb.org/file/REFERENCE.csv
wget http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet1.zip
wget http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet2.zip
wget http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet3.zip
unzip TrainingSet1.zip
unzip TrainingSet2.zip
unzip TrainingSet3.zip
cd ..
python util/convert_ICBEB.py
cp data/scp_statements.csv data/ICBEB/
#make sure the file order as same as the paper
cp data/icbeb_database.csv data/ICBEB/

####################################
#   GET PTBXL DATABASE
####################################

cd ECG-data
mkdir -p PTBXL
wget https://storage.googleapis.com/ptb-xl-1.0.1.physionet.org/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
#wget https://physionet.org/files/ptb-xl/1.0.1/ptbxl_database.csv
unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
cp -r ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/* PTBXL/
rm -r ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1
rm ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip