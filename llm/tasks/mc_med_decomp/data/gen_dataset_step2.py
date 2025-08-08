import pandas as pd
import os
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

def filter_dataframe_by_csn_time_df(df: pd.DataFrame, csn_time_df: pd.DataFrame) -> pd.DataFrame:
  """
  Filters a DataFrame, keeping rows for each CSN only if their
  'TimeDifferenceInMinutes' is less than or equal to a specified time limit
  provided in a separate DataFrame. Rows with CSNs not present in the
  csn_time_df are always kept.

  Args:
    df (pd.DataFrame): The input DataFrame to be filtered, with at least
                       'CSN' and 'TimeDifferenceInMinutes' columns.
    csn_time_df (pd.DataFrame): A DataFrame containing CSN values and their
                                corresponding time limits, in minutes. Expected columns are
                                'CSN' and 'TimeLimit'. Each CSN should ideally
                                appear only once.

  Returns:
    pd.DataFrame: The filtered DataFrame with index reset.
  Raises:
    ValueError: If 'CSN' or 'TimeLimit' columns are missing in csn_time_df.
  """
  if 'CSN' not in csn_time_df.columns or 'TimeLimit' not in csn_time_df.columns:
      raise ValueError("The 'csn_time_df' DataFrame must contain 'CSN' and 'TimeLimit' columns.")


  # Create a Series mapping CSN to TimeLimit for efficient lookup
  # .set_index('CSN') makes CSN the index, and ['TimeLimit'] selects the column of values
  time_limit_mapping = csn_time_df.set_index('CSN')['TimeLimit']

  # Get the list of CSNs that have a specific time limit
  csns_with_limits = csn_time_df['CSN'].tolist()

  # Create a boolean mask for filtering
  # Keep rows where:
  # 1. CSN is in the list of CSNs with specific time limits OR
  # 2. CSN IS in the list AND TimeDifferenceInMinutes is <= the mapped time limit

  mask_keep = (
      (df['CSN'].isin(csns_with_limits)) &
      (df['TimeDifferenceInMinutes'] <= df['CSN'].map(time_limit_mapping))
  )

  # Apply the mask to filter the DataFrame and reset the index
  filtered_df = df[mask_keep].reset_index(drop=True)

  return filtered_df

# --- Constants ---
CSN_COL = 'CSN'
LABEL_COL = 'Label'
TRIGGER_TIME_COL = 'Trigger_time'
SAMPLE_TIME_COL = 'Sample_time'
TIME_LIMIT_COL = 'TimeLimit'
TIME_DIFFERENCE_MINUTES_COL = 'TimeDifferenceInMinutes'
TEXT_COL = 'Text'
TIMEUPTO_COL = 'TimeUpto'

CONDENSED_TEXT_LABS_VITALS_MEDS_COL = 'Condensed_Text_Labs_Vitals_Meds'
CONDENSED_TEXT_VISITS_COL = 'Condensed_Text_Visits'
FINAL_TEXT_COL = 'Text' # Output column name

# Columns for visit condensation
VISIT_INFO_COLS = [
    'Age', 'Gender', 'Race', 'Means_of_arrival', 'Triage_Temp', 'Triage_HR',
    'Triage_RR', 'Triage_SBP', 'Triage_DBP', 'Triage_SPO2', 'Triage_acuity', 'CC'
]

# Separators for text condensation
NOTES_SEPARATOR = '\n'
VISIT_INFO_SEPARATOR = ', '
PATIENT_INFO_HEADER = "===Patient Info===\n"
LVM_HEADER = "===Labs, Vitals, Medications===\n"

def _filter_frequent_vitals(df, vital_events=['HR', 'SBP', 'DBP', 'RR', 'SpO2', 'MAP', 'Perf', '1min_HRV', '5min_HRV'], threshold=5, window_minutes=10):
    # Create a copy of the dataframe
    filtered_df = df.copy()
    
    # Extract event types from the Text column
    filtered_df['EventType'] = filtered_df['Text'].apply(lambda x: x.split(':')[0].strip())
    
    # Check which vitals occur more than the threshold times
    event_counts = filtered_df['EventType'].value_counts()
    frequent_vitals = [event for event in vital_events if event in event_counts and event_counts[event] > threshold]
    
    if not frequent_vitals:
        # print("No vital signs exceed the frequency threshold.")
        return df  # Return original dataframe if no filtering needed
    
    # Instead of trying to keep track of indices, we'll mark rows to keep or drop
    # Add a column to track which rows to keep
    filtered_df['Keep'] = True
    
    # For each vital that needs filtering
    for vital in frequent_vitals:
        # Get rows containing this vital
        vital_mask = filtered_df['EventType'] == vital
        vital_rows = filtered_df[vital_mask].copy()
        
        if vital_rows.empty:
            continue
            
        # Sort the vital's rows by time
        vital_rows = vital_rows.sort_values(TIME_DIFFERENCE_MINUTES_COL)
        
        # Mark all rows of this vital type for removal by default
        filtered_df.loc[vital_mask, 'Keep'] = False
        
        # Process in time windows
        current_window_start = vital_rows[TIME_DIFFERENCE_MINUTES_COL].iloc[0]
        kept_in_window = False
        
        for idx, row in vital_rows.iterrows():
            # If we're in a new time window, reset the flag
            if row[TIME_DIFFERENCE_MINUTES_COL] >= current_window_start + window_minutes:
                current_window_start = row[TIME_DIFFERENCE_MINUTES_COL]
                kept_in_window = False
                
            # Keep the first reading in each window
            if not kept_in_window:
                filtered_df.loc[idx, 'Keep'] = True
                kept_in_window = True

                # parse the value from the text and use 1 decimal place instead
                text = filtered_df.loc[idx, TEXT_COL]
                name, value = text.split(':')
                float_value = float(value)
                text = f"{name}: {float_value:.1f}"
                filtered_df.loc[idx, TEXT_COL] = text           
        
            # always add the last reading in the window
            if idx == vital_rows.index[-1]:
                filtered_df.loc[idx, 'Keep'] = True


    # Keep only the rows marked to keep
    result_df = filtered_df[filtered_df['Keep']].drop(columns=['EventType', 'Keep'])
    
    # Return the filtered dataframe
    return result_df.reset_index(drop=True)


def _condense_notes_per_csn_prefixes(group: pd.DataFrame) -> str:
    """Helper to condense text notes for a CSN group."""
    # We want to ensure that vitals, if frequent, are repeated only at 5 minute intervals
    # We can see if an event is a vital if it contains "HR|SBP|DBP|RR|SpO2|MAP' in the Text column

    # Filter out frequent vitals
    group = _filter_frequent_vitals(group, threshold=5, window_minutes=10)

    # For rows with the same time_difference, we want to combine them into a single entry
    # We can do this by grouping by time_difference and joining the text
    # We can use the groupby method to group by time_difference and join the text
    group = group.groupby(TIME_DIFFERENCE_MINUTES_COL).agg({
        TEXT_COL: lambda x: ";".join(x.astype(str))
    }).reset_index()


    # Sort the group by time_difference
    group = group.sort_values(by=TIME_DIFFERENCE_MINUTES_COL).reset_index(drop=True)

    formatted_entries = [{
            "time": row[TIME_DIFFERENCE_MINUTES_COL], 
            "text": f"{row[TIME_DIFFERENCE_MINUTES_COL]} - {row[TEXT_COL]}"
        } for _, row in group.astype({TEXT_COL: str, TIME_DIFFERENCE_MINUTES_COL: str}).iterrows()
    ]

    # Compute the list of dicts of time and cumulative prefixes
    # We iterate through the formatted entries, building the current prefix
    # by adding the next entry with the NOTES_SEPARATOR.
    cumulative_prefixes: List[Dict[str, any]] = []
    current_prefix = ""

    for i, entry in enumerate(formatted_entries):
        if i == 0:
            # The first prefix is just the first entry
            current_prefix = entry['text']
        else:
            # Subsequent prefixes append the new entry with the separator
            current_prefix = current_prefix + NOTES_SEPARATOR + entry['text']
        cumulative_prefixes.append({ 'time': entry['time'], 'text': current_prefix } )

    return cumulative_prefixes


def _condense_notes_per_csn(group: pd.DataFrame) -> str:
    """Helper to condense text notes for a CSN group."""
    # Ensure that frequent vitals are repeated only at 5 minute intervals
    group = _filter_frequent_vitals(group, threshold=5, window_minutes=10)

    # For rows with the same time_difference, we want to combine them into a single entry
    group = group.groupby(TIME_DIFFERENCE_MINUTES_COL).agg({
        TEXT_COL: lambda x: ";".join(x.astype(str))
    }).reset_index()

    return NOTES_SEPARATOR.join(
        f"{row[TIME_DIFFERENCE_MINUTES_COL]} - {row[TEXT_COL]}"
        for _, row in group.astype({TEXT_COL: str, TIME_DIFFERENCE_MINUTES_COL: str}).iterrows()
    )

def _condense_visit_info_per_csn(group: pd.DataFrame) -> pd.Series:
    """Helper to condense visit information for a CSN group, using the first row."""
    first_row = group.iloc[0]
    csn = first_row[CSN_COL]
    
    condensed_text_parts = []
    for col in VISIT_INFO_COLS:
        if col in first_row and pd.notna(first_row[col]):
            condensed_text_parts.append(f"{col}: {first_row[col]}")
            
    return pd.Series({
        CSN_COL: csn,
        CONDENSED_TEXT_VISITS_COL: VISIT_INFO_SEPARATOR.join(condensed_text_parts)
    })

def _process_csn_group(group):
    """
    Processes a single CSN group to condense notes and extract the label.
    Assumes the group is a pandas DataFrame for a single CSN.
    """
    csn = group[CSN_COL].iloc[0]
    condensed_text = _condense_notes_per_csn(group)
    # Get the label for this CSN (all rows in group should have same label)
    label = group[LABEL_COL].iloc[0] if LABEL_COL in group else None

    return {
            CSN_COL: csn,
            CONDENSED_TEXT_LABS_VITALS_MEDS_COL: condensed_text,
            LABEL_COL: label
        } 

def _process_csn_group_stability(group):
    """
    Processes a single CSN group to condense notes and extract the label.
    Assumes the group is a pandas DataFrame for a single CSN.
    """
    csn = group[CSN_COL].iloc[0]
    condensed_text_list = _condense_notes_per_csn_prefixes(group)
    # Get the label for this CSN (all rows in group should have same label)
    label = group[LABEL_COL].iloc[0] if LABEL_COL in group else None

    return [
        {
            CSN_COL: csn,
            TIMEUPTO_COL: entry['time'],
            CONDENSED_TEXT_LABS_VITALS_MEDS_COL: entry['text'],
            LABEL_COL: label
        } for entry in condensed_text_list
    ]

def condense_time_series_text_data(
    filtered_notes_df: pd.DataFrame,
    stability: bool = False, # If True, use _condense_notes_per_csn_prefixes
    max_workers: int = None # Allow specifying max workers, default to os.cpu_count()
) -> pd.DataFrame:
    """
    Condenses time-series text data (Labs, Vitals, Meds) per case number (CSN).

    Args:
        filtered_notes_df: DataFrame containing time-series text data with at least
                           CSN_COL, 'Text', and 'TimeDifferenceInMinutes' columns.
                           May also contain LABEL_COL.
        max_workers: The maximum number of worker processes to use for parallelization. If None, defaults to the number of CPU cores.

    Returns:
        A DataFrame with condensed text per CSN, including the CSN and optionally the label.
    """
    print(f"Starting condensation of time-series text data for {len(filtered_notes_df)} rows...")

    if not filtered_notes_df.empty:
        # Get the list of groups
        csn_groups = [group for _, group in filtered_notes_df.groupby(CSN_COL)]

        # Determine the number of workers
        if max_workers is None:
           max_workers = os.cpu_count()
           print(f"Using default max_workers: {max_workers}")
        else:
            print(f"Using specified max_workers: {max_workers}")


        # Parallelize the processing of groups using process_map with tqdm
        # The _process_csn_group function is applied to each group in csn_groups
        # Ensure chunksize is at least 1 to avoid errors with small numbers of groups
        chunksize = max(1, len(csn_groups) // max_workers)
        print(f"Using chunksize: {chunksize}")

        condensed_notes_list = process_map(
            _process_csn_group_stability if stability else _process_csn_group,
            csn_groups,
            max_workers=max_workers,
            chunksize=chunksize,
            desc="Condensing Notes", # Description for the tqdm progress bar
            # Added allow_fork=True and unit='group' for better progress bar and compatibility
            # allow_fork=True, # May be needed depending on OS and context
            unit='group'
        )

        # Flatten the list of lists of dictionaries into a single list of dictionaries
        if stability:
            # If using _condense_notes_per_csn_prefixes, we need to flatten the list of lists
            condensed_notes_list = [item for sublist in condensed_notes_list for item in sublist]
            condensed_notes_list = [item for item in condensed_notes_list if item is not None]
        condensed_notes_df = pd.DataFrame(condensed_notes_list)
    else:
        print("Input DataFrame is empty. Returning empty DataFrame.")
        # Ensure the columns match the potential output columns
        columns = [CSN_COL, CONDENSED_TEXT_LABS_VITALS_MEDS_COL]
        if LABEL_COL:
            columns.append(LABEL_COL)
        condensed_notes_df = pd.DataFrame(columns=columns)

    print(f"Condensed notes DataFrame has {len(condensed_notes_df)} rows")
    return condensed_notes_df


def generate_textified_dataset(
    final_df: pd.DataFrame,
    mc_data_path: str,
    # horizon_time_minutes: int = 90,
    label_file_path: str = 'eSOFA.parquet',
    visits_file_name: str = 'visits.csv',
    out_file_path: str = 'textified_dataset.parquet',
    min_horizon_time_minutes: int = 30,
    max_horizon_time_minutes: int = 150
) -> pd.DataFrame:
    """
    Generates a dataset with condensed textual information per CSN.

    The process involves:
    1. Loading label data and calculating a 'TimeLimit' for filtering.
       - 'TimeLimit' is calculated based on 'Trigger_time' or 'Sample_time',
         assuming these are in hours and horizon_time_minutes is in minutes.
         The formula is: (time_in_hours * 60) - horizon_time_minutes.
    2. Filtering the input 'final_df' based on CSN and this 'TimeLimit'.
    3. Merging 'Label' information into the filtered data.
    4. Condensing time-series text data (from 'final_df') per CSN.
       - Format: "TimeDifferenceInMinutes: Text; TimeDifferenceInMinutes: Text..."
    5. Loading and condensing patient visit information per CSN from 'visits.csv'.
       - Format: "Age: Value, Gender: Value..."
    6. Merging the condensed time-series data with condensed visit data.
    7. Combining both condensed text fields into a single 'Text' column.

    Args:
        final_df: DataFrame containing time-series data including at least
                  ['CSN', 'TimeDifferenceInMinutes', 'Text'].
        mc_data_path: Path to the directory containing visit file.
        horizon_time_minutes: Time in minutes to define the observation window
                              limit before a trigger or sample time.
        label_file_name: Name of the Parquet file containing labels and trigger/sample times.
        visits_file_name: Name of the CSV file containing visit information.

    Returns:
        A DataFrame with one row per CSN, containing 'CSN', 'Label', and a
        combined 'Text' field with patient info and time-series data.
        
    Requires:
        An external function `filter_dataframe_by_csn_time_df(df, criteria_df)` 
        to be defined and available in the scope.
    """
    # --- 1. Load and Prepare Labels ---
    labels_df = pd.read_parquet(label_file_path)
    print(f"Labels loaded from {label_file_path}")
    print("Number of labels:", len(labels_df))

    # Lets compute notes for intervening time
    upto_trigger_labels = labels_df.copy()
    upto_trigger_labels[TIME_LIMIT_COL] = upto_trigger_labels[TRIGGER_TIME_COL].fillna(upto_trigger_labels[SAMPLE_TIME_COL]) * 60
    upto_trigger_time_notes = filter_dataframe_by_csn_time_df(final_df, upto_trigger_labels)


    # Calculate TimeLimit.
    # Logic: labels['TimeLimit'] = labels['Trigger_time'].fillna(labels['Sample_time']) * 60 - horizon_time
    time_for_limit = labels_df[TRIGGER_TIME_COL].fillna(labels_df[SAMPLE_TIME_COL])
    labels_df[TIME_LIMIT_COL] = time_for_limit * 60 - min_horizon_time_minutes
    print(f"TimeLimit calculated for {len(labels_df)} labels")
    # remove negative time limits
    labels_df = labels_df[labels_df[TIME_LIMIT_COL] >= 0]
    print(f"Filtered labels DataFrame to {len(labels_df)} rows with non-negative TimeLimit")
    
    # --- 2. Filter main DataFrame ---
    # `filter_dataframe_by_csn_time_df` is an external function.
    # It needs the main DataFrame and a criteria DataFrame (CSN and TimeLimit).
    # We assume `final_df` has `TIME_DIFFERENCE_MINUTES_COL` used for filtering against `TIME_LIMIT_COL`.
    time_criteria_df = labels_df[[CSN_COL, TIME_LIMIT_COL]]
    filtered_notes_df = filter_dataframe_by_csn_time_df(final_df, time_criteria_df)
    print(f"Filtered notes DataFrame has {len(filtered_notes_df)} rows after applying time criteria")
    # intervening time 
    remaining_notes_df = upto_trigger_time_notes.merge(filtered_notes_df, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    print(f"Intervening time DataFrame has {len(remaining_notes_df)} rows")

    # --- 3. Add Label column to filtered_df ---
    # Keep only relevant columns from labels_df for merging
    labels_to_merge = labels_df[[CSN_COL, LABEL_COL]].drop_duplicates(subset=[CSN_COL])
    filtered_notes_df = pd.merge(filtered_notes_df, labels_to_merge, on=CSN_COL, how='left')
    remaining_notes_df = pd.merge(remaining_notes_df, labels_to_merge, on=CSN_COL, how='left')
    print(f"Labels merged into filtered notes DataFrame, now has {len(filtered_notes_df)} rows")
    print(f"Labels merged into remaining notes DataFrame, now has {len(remaining_notes_df)} rows")

    # --- 4. Condense Time-Series Text Data (Labs, Vitals, Meds) ---
    condensed_notes_df = condense_time_series_text_data(filtered_notes_df)
    condensed_remaining_notes_df = condense_time_series_text_data(remaining_notes_df)
    condensed_remaining_notes_df.rename(columns={CONDENSED_TEXT_LABS_VITALS_MEDS_COL: FINAL_TEXT_COL}, inplace=True)


    # --- 5. Load and Condense Visit Information ---
    visits_file_path = os.path.join(mc_data_path, visits_file_name)
    # Specify dtype for CSN to handle potential mixed types or leading zeros if CSN is numeric string
    visits_df = pd.read_csv(visits_file_path, dtype={CSN_COL: 'Int64'}) # or str if appropriate

    # Keep only visits for CSNs present in our filtered data
    relevant_csns = filtered_notes_df[CSN_COL].unique()
    visits_filtered_df = visits_df[visits_df[CSN_COL].isin(relevant_csns)]

    if not visits_filtered_df.empty:
        # Taking the first row if multiple visit entries exist for a CSN
        condensed_visits_df = visits_filtered_df.groupby(CSN_COL, as_index=False).apply(_condense_visit_info_per_csn)
    else:
        condensed_visits_df = pd.DataFrame(columns=[CSN_COL, CONDENSED_TEXT_VISITS_COL])
    print(f"Condensed visits DataFrame has {len(condensed_visits_df)} rows")
        
    # --- 6. Merge Condensed Data ---
    # Merge condensed notes (which now includes Label) with condensed visits
    if not condensed_notes_df.empty and not condensed_visits_df.empty:
        merged_condensed_df = pd.merge(condensed_notes_df, condensed_visits_df, on=CSN_COL, how='left')
        # Fill NA for visits text if a CSN had notes but no visit info after filtering
        merged_condensed_df[CONDENSED_TEXT_VISITS_COL] = merged_condensed_df[CONDENSED_TEXT_VISITS_COL].fillna('')
    elif not condensed_notes_df.empty: # No visits data to merge
        merged_condensed_df = condensed_notes_df
        merged_condensed_df[CONDENSED_TEXT_VISITS_COL] = '' # Add empty column for consistency
    else: # No notes data, potentially no data at all
        # This case means filtered_notes_df was empty. Result should be empty with correct columns.
        return pd.DataFrame(columns=[CSN_COL, LABEL_COL, FINAL_TEXT_COL])
    print(f"Merged condensed DataFrame has {len(merged_condensed_df)} rows")


    # --- 7. Combine Text Fields ---
    # Ensure parts are strings before concatenation
    merged_condensed_df[CONDENSED_TEXT_VISITS_COL] = merged_condensed_df[CONDENSED_TEXT_VISITS_COL].astype(str)
    merged_condensed_df[CONDENSED_TEXT_LABS_VITALS_MEDS_COL] = merged_condensed_df[CONDENSED_TEXT_LABS_VITALS_MEDS_COL].astype(str)

    merged_condensed_df[FINAL_TEXT_COL] = (
        f"{PATIENT_INFO_HEADER}\n" +
        merged_condensed_df[CONDENSED_TEXT_VISITS_COL] +
        f"\n{LVM_HEADER}\n" + # Added extra separator for clarity like original
        merged_condensed_df[CONDENSED_TEXT_LABS_VITALS_MEDS_COL]
    )

    # --- 8. Filter for TIMEUPTO_COL > MAX HORIZON TIME --- 
    # max_times = labels_df.copy()
    # max_times[TIME_LIMIT_COL] = time_for_limit * 60 - max_horizon_time_minutes
    # # merge max_times with merged_condensed_df
    # print("Number of unique CSNs in merged_condensed_df:", len(merged_condensed_df[CSN_COL].unique()))
    # merged_condensed_df = pd.merge(merged_condensed_df, max_times[[CSN_COL, TIME_LIMIT_COL]], on=CSN_COL, how='left')
    # merged_condensed_df[TIMEUPTO_COL] = merged_condensed_df[TIMEUPTO_COL].astype(float)
    # merged_condensed_df = merged_condensed_df[merged_condensed_df[TIMEUPTO_COL] >= merged_condensed_df[TIME_LIMIT_COL]]
    # print("Number of unique CSNs in merged_condensed_df:", len(merged_condensed_df[CSN_COL].unique()))

    # --- 9. Final Selection and Return ---
    # Keep CSN, Label (already in merged_condensed_df from condensed_notes_df), and the final Text
    result_df = merged_condensed_df[[CSN_COL, LABEL_COL, FINAL_TEXT_COL]]
    remaining_df = condensed_remaining_notes_df[[CSN_COL, LABEL_COL, FINAL_TEXT_COL]]
    
    # --- 10. Write to file ---
    # Save the final DataFrame to a Parquet file
    out_file_path = os.path.join(out_file_path)
    # result_df.to_parquet(out_file_path, index=False)
    remaining_df.to_parquet(out_file_path, index=False)
    print(f"Textified dataset saved to {out_file_path}")

if __name__ == "__main__":
    # argparse
    import argparse
    parser = argparse.ArgumentParser(description="Generate a textified dataset from medical data.")
    parser.add_argument('--final_df_path', type=str, help="Path to the final DataFrame.", default='~/projects/arpa-h/data_prep/mc-med/intermediate/final_df.parquet')
    parser.add_argument('--mc_data_path', type=str, help="Path to the directory containing visit files.", default='~/projects/mc-med/mc-med-1.0.0/data/')
    parser.add_argument('--label_file_path', type=str, help="Path to the label file.", default='~/projects/mc-med/data/eSOFA.parquet')
    parser.add_argument('--out_file_path', type=str, help="Output file path for the textified dataset.", default='~/projects/arpa-h/llm/tasks/mc_med_sepsis/data/remaining.parquet')

    args = parser.parse_args()

    generate_textified_dataset(
        final_df=pd.read_parquet(args.final_df_path),
        mc_data_path=args.mc_data_path,
        out_file_path=args.out_file_path,
        label_file_path=args.label_file_path,
        min_horizon_time_minutes=90,
        # max_horizon_time_minutes=150,
    )

    
