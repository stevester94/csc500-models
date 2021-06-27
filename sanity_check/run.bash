#! /usr/bin/env bash
set -eou pipefail

source_dataset_base_path="$CSC500_ROOT_PATH/datasets/automated_windower"
results_base_path="$CSC500_ROOT_PATH/csc500-past-runs/maximal_convolution/"

original_batch_size=100
patience=10


for desired_batch_size in 512; do
for epochs in 10; do
for learning_rate in 0.0001; do
for distance in "2.8.14.20.26.32"; do
for seed in 906  2856 3007 7701
    experiment_name=deeper.no.tipoff_seed-${seed}_distance-${distance}_learningRate-${learning_rate}_batch-${desired_batch_size}_epochs-${epochs}_patience-$patience
    echo "Begin $experiment_name" | tee logs
    rm -rf best_weights model_checkpoint *png checkpoint logs experiment_name results.csv training_log.csv details.txt
    cat << EOF | python3 ./conv.py 2>&1 | tee --append logs
    {
        "seed": $seed,
        "experiment_name": "$experiment_name",
        "source_dataset_path": "$source_dataset_base_path/windowed_EachDevice-200k_batch-100_stride-20_distances-$distance",
        "learning_rate": $learning_rate,
        "original_batch_size": $original_batch_size,
        "desired_batch_size": $desired_batch_size,
        "epochs": $epochs,
        "patience": $patience
    }
EOF

    cp -R . $results_base_path/$experiment_name
    rm $results_base_path/$experiment_name/.gitignore

done
done
done
done
