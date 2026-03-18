#!/bin/bash

# Define the range and step
start=0.0
end=1
step=0.1

# Loop through alpha and beta values
for alpha in $(seq $start $step $end); do
  for beta in $(seq $start $step $end); do
    name="a_${alpha}_b_${beta}"
    echo "Running: python main.py --name $name --alpha $alpha --beta $beta"
    python main.py --name $name --alpha $alpha --beta $beta --wandb --device 0
  done
done