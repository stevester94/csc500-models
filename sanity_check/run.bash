#! /usr/bin/env bash
set -eou pipefail

source_dataset_base_path="$CSC500_ROOT_PATH/datasets/automated_windower"
results_base_path="$CSC500_ROOT_PATH/csc500-past-runs/maximal_convolution/"

original_batch_size=100
patience=10



for desired_batch_size in 512; do
for epochs in 300; do
for learning_rate in 0.0001; do
for distance in "2.8.14.20.26.32" 16; do
# for seed in 8646 25792 15474 5133 30452 17665 27354 17752 3854 17536 8272 14591 10045 22635 14858 18363 16886 26584 10365 12026 31946 7292 2523 28811 11117 22598 22071 7765 159 7053 21385 8487 6991 28269 7990 26763 24144 6842 31160 12236 18720 19404 18531 4804 17757 31853 1730 22815 24494 14324 22963 10736 1664 1313 5151 13851 29403 9744 14533 14578 9129 9561 9837 23068 2101 9866 22899 28828 25714 26435 30176 24068 295 4515 7340 15521 5630 21544 9794 6006 24538 13508 14815 3473 8435 5021 27950 11295 28643 5384 4550 10220 14759 12332 21133 22659 5729 16450 31710 15005; do
for seed in 306; do
    experiment_name=yolo_deeper_tipoff-no_seed-${seed}_distance-${distance}_learningRate-${learning_rate}_batch-${desired_batch_size}_epochs-${epochs}_patience-$patience
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
done