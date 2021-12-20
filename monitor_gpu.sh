#!/bin/bash
while sleep 1; do
  temperatures=$(nvidia-smi --format=csv,noheader --query-gpu=temperature.gpu)
  for temperature in $temperatures
  do
    echo 'GPU temperature:' $temperature
    if (( $temperature>90 )); then
	  pids=$(nvidia-smi | grep 'python' | awk '{ print $3 }')
	  for pid in $pids
	  do
	    kill -9 $pid
	  done
	  exit 1
	fi
  done
  echo '---'
done