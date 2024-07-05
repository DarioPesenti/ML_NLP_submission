activations=("tanh" "relu")
learning_rates=(0.0001 0.001 0.01 0.1)
regularizations=(0.0001 0.001 0.01 0.1 1)
hidden_dims=(2 4 8 16 32)

for activation in "${activations[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for reg in "${regularizations[@]}"; do
      for hdim in "${hidden_dims[@]}"; do
        python nn_from_scratch.py \
          --lr $lr \
          --reg $reg \
          --activation $activation \
          --hdim $hdim \
          --wandb
      done
    done
  done
done
