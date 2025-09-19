#%%
import os
import json
import pandas as pd
import re


def load_fid_metrics(root_dir: str) -> pd.DataFrame:
    """
    Iterate through subfolders, read data files, and return a DataFrame where 
    each row contains folder name (with subdata), FID values, actual_kimg, and 
    all training_options.json values.
    """
    rows = []

    for entry in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, entry)
        jsonl_path = os.path.join(folder_path, "metric-fid50k_full.jsonl")
        training_options_path = os.path.join(folder_path, "training_options.json")
        stats_path = os.path.join(folder_path, "stats.jsonl")
        # log_txt_path = os.path.join(folder_path, "log.txt")

        # Only process if it's a directory and the JSONL file exists
        if os.path.isdir(folder_path) and os.path.isfile(jsonl_path) and os.path.isfile(training_options_path):
            # Load training options
            with open(training_options_path, "r") as f:
                training_options = json.load(f)

            # Load FID values
            fid_values = []
            with open(jsonl_path, "r") as f:
                for line in f:
                    record = json.loads(line)
                    fid = record.get("results", {}).get("fid50k_full")
                    if fid is not None:
                        fid_values.append(fid)

            # Calculate actual_kimg (runs would frequently crash / finish early)
            actual_kimg = 40 * (len(fid_values) - 1)

            # Calculate minimum FID value
            min_fid = min(fid_values) if fid_values else None

            # Extract all subdata from training_set_kwargs and data_loader_kwargs
            ts_kwargs = training_options.get("training_set_kwargs", {})
            dl_kwargs = training_options.get("data_loader_kwargs", {})

            # Format subheadings and values as strings
            ts_subdata = ", ".join([f"{k}={v}" for k, v in ts_kwargs.items()])
            dl_subdata = ", ".join([f"{k}={v}" for k, v in dl_kwargs.items()])

            folder_name = entry  # Just the folder name

            # Add subdata as separate columns
            ts_subdata_dict = {f"ts_{k}": v for k, v in ts_kwargs.items()}
            dl_subdata_dict = {f"dl_{k}": v for k, v in dl_kwargs.items()}

            # Extract avg_sec_kimg and run_duration from stats.jsonl
            timing_sec_per_kimg_list = []
            timing_total_hours_list = []
            if os.path.isfile(stats_path):
                with open(stats_path, "r") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if "Timing/sec_per_kimg" in record:
                                timing_sec_per_kimg_list.append(record["Timing/sec_per_kimg"].get("mean"))
                            if "Timing/total_hours" in record:
                                timing_total_hours_list.append(record["Timing/total_hours"].get("mean"))
                        except Exception:
                            continue
            avg_sec_kimg = (sum(timing_sec_per_kimg_list) / len(timing_sec_per_kimg_list)) if timing_sec_per_kimg_list else None
            run_duration = max(timing_total_hours_list) if timing_total_hours_list else None

            row = {
                "folder": folder_name,
                "fid_values": fid_values,
                "actual_kimg": actual_kimg,
                "min_fid": min_fid,
                "avg_sec_kimg": avg_sec_kimg,
                "run_duration": run_duration,
            }
            row.update(training_options)
            row.update(ts_subdata_dict)
            row.update(dl_subdata_dict)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df

#%%

df_fid = load_fid_metrics(r".\Training_ANDA\training-runs")
with pd.ExcelWriter("output_results.xlsx") as writer:
    df_fid.to_excel(writer, sheet_name="fid_metrics", index=False)
