import pandas as pd
import glob
# Import ProcessPoolExecutor instead of ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import os
import time # Added to potentially track processing time per file

# Load labels and d_items outside the processing function as they are used for all files
try:
    labels = pd.read_csv('mortality_labels.csv')
    print("Loaded mortality_labels.csv")
except FileNotFoundError:
    print("Error: mortality_labels.csv not found. Please ensure it's in the same directory as the script.")
    exit()

try:
    # Assuming the d_items path is correct for your system
    d_items = pd.read_csv('/home/mkeoliya/projects/arpa-h/MIMIC/mimiciv/3.1/icu/d_items.csv')
    print("Loaded d_items.csv")
except FileNotFoundError:
    print("Error: d_items.csv not found. Please ensure it's in the correct path.")
    exit()

# Data preprocessing steps from the notebook
labels['Time'] = pd.to_datetime(labels['deathtime'].combine_first(labels['outtime'])) - pd.Timedelta(hours=12)
d_items = d_items[['itemid', 'abbreviation']]

def process_parquet_file(file_path):
    """Processes a single parquet file based on the logic from the notebook."""
    start_time = time.time()
    print(f"Processing {file_path} in process {os.getpid()}...")
    try:
        # Removed extra print statements inside to reduce output spam
        final_df = pd.read_parquet(file_path)
        # print("Final DataFrame shape:", final_df.shape) # Optional: keep for debugging if needed

        # merge labels with final_df
        merged_df = final_df.merge(labels[['subject_id', 'Time']], on=['subject_id'], how='inner')

        # convert 'time' in final_df to datetime and rename
        merged_df['event_time'] = pd.to_datetime(merged_df['time'])
        merged_df = merged_df.drop(columns=['time']) # Drop the original 'time' column

        # keep only the event_times before 'Time'
        merged_df = merged_df[merged_df['event_time'] <= merged_df['Time']]

        # two day cutoff calculation and filtering
        merged_df['StartTime'] = merged_df['Time'] - pd.Timedelta(days=1.5)
        merged_df = merged_df[merged_df['event_time'] >= merged_df['StartTime']]
        merged_df = merged_df.drop(columns=['Time', 'StartTime']) # Drop temporary time columns

        # The original notebook uses .head(10000) here, which might be for testing.
        # We will remove it to process the full data. If you need to limit the data, uncomment the line below:
        # merged_df = merged_df.head(10000)

        # pandas operations - these are CPU heavy
        merged_df_indexed = merged_df.set_index('event_time')

        # Resample and filter data
        # print("Resampling and filtering data...") # Optional: keep for debugging if needed
        # print("Merged DataFrame shape before resample:", merged_df_indexed.shape) # Optional: keep for debugging if needed

        # Note: The resample logic is kept as per your original code.
        # For large datasets, this can still be memory intensive per process.
        filtered_df = merged_df_indexed.groupby(['subject_id', 'itemid']).resample('30T').first().dropna()
        # Reset indices multiple times as in original logic
        filtered_df = filtered_df.reset_index(level=0, drop=True)
        filtered_df = filtered_df.reset_index(level=0, drop=True)
        filtered_df = filtered_df.reset_index()
        # print("Filtered DataFrame shape after resample:", filtered_df.shape) # Optional: keep for debugging if needed

        # get the last 500 rows of each subject_id
        # This requires sorting first to ensure 'tail(500)' gets the latest times
        filtered_df = filtered_df.sort_values(by=['subject_id', 'event_time'])
        filtered_df = filtered_df.groupby('subject_id').tail(700)


        # keep only relevant columns and calculate relative time
        filtered_df['relative_time_hrs'] = filtered_df.groupby('subject_id')['event_time'].transform(lambda x: (x - x.min()).dt.total_seconds() / 3600)
        filtered_df = filtered_df[['subject_id', 'itemid', 'value', 'relative_time_hrs']]

        # convert itemid to int
        filtered_df['itemid'] = filtered_df['itemid'].astype(int)

        # merge d_items with filtered_df
        # This merge might be slightly faster done *after* filtering/resampling
        # as filtered_df should be smaller than merged_df
        filtered_df = filtered_df.merge(d_items, on='itemid', how='inner')

        # create 'Text' column
        # Ensure 'value' is string before concatenating
        filtered_df['Text'] = filtered_df['abbreviation'] + ': ' + filtered_df['value'].astype(str)

        # sort by subject_id and relative_time_hrs (needed for the final condense step's order)
        filtered_df = filtered_df.sort_values(by=['subject_id', 'relative_time_hrs'])

        # Helper function remains the same, it operates on pandas DataFrames
        def _condense_notes_per_subject(group: pd.DataFrame) -> str:
            """Helper to condense text notes for a subject_id group."""
            # For rows with the same relative_time_hrs, we want to combine their Text
            # Ensure relative_time_hrs is rounded/consistent for grouping if needed, or group by the exact value
            group = group.groupby('relative_time_hrs').agg({
                'Text': lambda x: ";".join(x.astype(str))
            }).reset_index()

            # Sort by relative_time_hrs within the condensed group before joining lines
            group = group.sort_values(by='relative_time_hrs')

            return "\n".join(
                f"{row['relative_time_hrs']:.4f} - {row['Text']}" # Format time for consistency
                for _, row in group.astype({'Text': str}).iterrows() # relative_time_hrs is now float
            )

        # group by subject_id and condense the text notes
        # This apply is still a bottleneck but will run in parallel processes
        print(f"Condensing text for {file_path}...")
        condensed_df = filtered_df.groupby('subject_id').apply(_condense_notes_per_subject).reset_index()
        condensed_df.rename(columns={0: 'Text'}, inplace=True)
        print(f"Finished condensing text for {file_path}. Condensed shape: {condensed_df.shape}")


        # Save the processed data.
        output_filename = os.path.splitext(file_path)[0] + '_processed.parquet'
        condensed_df.to_parquet(output_filename, index=False) # Added index=False as subject_id is already a column
        end_time = time.time()
        print(f"Finished processing {file_path} in process {os.getpid()}. Saved to {output_filename}. Time taken: {end_time - start_time:.2f} seconds.")

    except Exception as e:
        print(f"Error processing {file_path} in process {os.getpid()}: {e}")

# Find all {i}_final.parquet files
parquet_files = glob.glob('*_final.parquet')

if not parquet_files:
    print("No files matching '*_final.parquet' found in the current directory.")
else:
    print(f"Found {len(parquet_files)} files to process.")
    # Use ProcessPoolExecutor for multiprocessing
    # Adjust max_workers based on your system's number of CPU cores and available memory
    # Using os.cpu_count() is a good default.
    # with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    #     # executor.map runs the function for each item in the iterable
    #     # It returns results in the order the tasks were submitted (though not used here)
    #     # Errors raised in the function will be re-raised by map upon iteration
    #     list(executor.map(process_parquet_file, parquet_files)) # Consume the iterator to ensure tasks complete

    print("All file processing tasks submitted. Waiting for completion...")

    # The 'with' block ensures all processes finish before proceeding

    print("All individual files processed.")

    # load all the files and concatenate them into a single dataframe
    # Ensure you only glob for the *processed* files after the multiprocessing is done
    processed_files = glob.glob("*_final_processed.parquet")
    if processed_files:
        print(f"Concatenating {len(processed_files)} processed files...")
        try:
            # It might be slightly more memory efficient to read and concatenate in chunks if you have *many* processed files
            # But for a reasonable number, pandas.concat is fine.
            df = pd.concat([pd.read_parquet(file) for file in processed_files], ignore_index=True)
            df.to_parquet("all_final_processed.parquet", index=False)
            print(f"Concatenated data saved to all_final_processed.parquet. Final shape: {df.shape}")
        except Exception as e:
             print(f"Error during concatenation: {e}")
    else:
        print("No processed files found to concatenate.")


print("Overall process finished.")