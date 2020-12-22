#!/bin/bash
while sleep 1; do
  temperature=$(nvidia-smi --format=csv,noheader --query-gpu=temperature.gpu)
  echo $temperature
  if (( $temperature>90 )); then
    pid=$(nvidia-smi | grep 'python' | awk '{ print $3 }')
    kill -9 $pid
    break
  fi
done