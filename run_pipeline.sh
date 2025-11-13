set -xe

# This is just to illustrate the order in which the scripts were run
# to generate the training/validation/testing data and has not been tested

python shuffle_datasets.py

python apply_filters.py
rm -rf data/wikipedia_shuffled/
mv data/wikipedia_shuffled_filtered/ data/wikipedia_shuffled/

python add_lang_column.py
rm -rf data/wikipedia_shuffled/
mv data/wikipedia_shuffled_filtered/ data/wikipedia_shuffled/

python create_splits.py
rm -rf data/wikipedia_shuffled/
mv data/wikipedia_shuffled_split/ data/wikipedia_shuffled/

python create_training_data.py

python generate_datasets.py
