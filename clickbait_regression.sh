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
python /home/emperor/scripts/jsonl_to_dataframe.py -i $INPUT -d /home/emperor/data_dir/

echo "Running Text CNN"
python /home/emperor/scripts/load_text_cnn_script.py -m /home/emperor/model/model -d /home/emperor/data_dir/ -o $OUTPUT
