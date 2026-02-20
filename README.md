## Run the following command to generate the data dump in the data/raw/ directory

``` sudo echo "Successfully Authenticated"
sudo python3 collect_training_data.py ```

## Run the following command to form the merged dataset from the 4 .jsonl files

> python merge_dataset.py \ </br>
>   --state data/raw/state_20260220_010514.jsonl \ </br>
>   --pmc data/raw/pmc_20260220_010514.jsonl \ </br>
>   --rapl data/raw/rapl_20260220_010514.jsonl \ </br>
>   --output data/processed/final_dataset.csv` </br>

## Run the following command to generate the graphs

> sudo echo "Successfully Authenticated" </br>