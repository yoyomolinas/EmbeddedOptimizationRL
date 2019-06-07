#!/usr/bin/env bash
i=33

# Analyze mov files
for filename in ../../Vids/test/*.mov; do
   echo $filename
   # python3 generate_data.py --video $filename --visualize --save_to data/$i.json
   python3 ../../multithreaded_run.py --video $filename --save_to data/$i.json --include_feature
   i=$((i+1))
done

# Analyze mp4 files
for filename in ../../Vids/test/*.mp4; do
   echo $filename
   # python3 generate_data.py --video $filename --visualize --save_to data/$i.json
   python3 ../../multithreaded_run.py --video $filename --save_to data/$i.json --include_feature
   i=$((i+1))
done
