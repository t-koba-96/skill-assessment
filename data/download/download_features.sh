#!/bin/bash
fileid="1kUB8K7f5OEeOnDA4qD5qqlGiflnK8cHm"
filename="EPIC-Skills_i3d_features.tar.gz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
mkdir -p EPIC-Skills/features/
tar -C EPIC-Skills/features/ -zxvf ${filename}

##BEST
fileid="1AB_ddG1yoQxSGHAeNMEQBp4A7TQFhTMm"
filename="BEST_i3d_features.tar.gz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
mkdir -p BEST/features/
tar -C BEST/features/ -zxvf ${filename}
