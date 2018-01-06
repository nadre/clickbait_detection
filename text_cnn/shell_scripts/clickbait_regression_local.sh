#!/usr/bin/env bash
while getopts i:o: option
do
 case "${option}"
 in
 i) INPUT=${OPTARG};;
 o) OUTPUT=${OPTARG};;
 esac
done

echo $INPUT
echo $OUTPUT

echo "Tokenizing JSONL"
python3.4 jsonl_to_dataframe.py -i $INPUT -d '/home/xuri3814/data/clickbait/data_dir/'

echo "Running Text CNN"
python3.4 load_text_cnn_script.py -m /home/xuri3814/data/clickbait/model/model -d '/home/xuri3814/data/clickbait/data_dir/' -o $OUTPUT
