#!/usr/bin/env python3
"""
Split master fMRI events file into run-specific BIDS events files.

This script:
1. Splits the master events file into 6 run-specific BIDS events files
2. Fixes heat stimulus rows with proper onset, duration, trial_type mapping
3. Adds/ensures required behavioral columns
4. Creates clean BIDS-compliant events files for GLM analysis

Usage:
    python split_events_to_runs.py [--input INPUT_FILE] [--output_dir OUTPUT_DIR] [--dry_run]
"""

import argparse
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys


def load_eeg_drop_log(drop_log_path):
    """Load EEG drop log if available."""
    drop_log_file = Path(drop_log_path)
    if not drop_log_file.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(drop_log_file, sep='\t')
        if 'run' in df.columns and 'trial_number' in df.columns:
            return df
        else:
            print(f"  Warning: Drop log missing 'run' or 'trial_number' columns")
            return pd.DataFrame()
    except Exception as e:
        print(f"  Warning: Could not load drop log: {e}")
        return pd.DataFrame()

def setup_temperature_mapping():
    """Define temperature to trial_type mapping (no dots in names)."""
    return {
        44.3: 'temp44p3',
        45.3: 'temp45p3', 
        46.3: 'temp46p3',
        47.3: 'temp47p3',
        48.3: 'temp48p3',
        49.3: 'temp49p3'
    }

def process_run_events(df_run, run_number, temp_mapping, include_delay=True):
    """
    Process events for a single run to create BIDS-compliant format.
    Includes heat stimuli, decision periods, rating periods, and optional delay periods.
    
    Parameters:
    -----------
    df_run : pd.DataFrame
        Events data for this run
    run_number : int
        Run number (1-6)
    temp_mapping : dict
        Mapping from temperature to trial_type names
    include_delay : bool
        Whether to include delay nuisance regressor
        
    Returns:
    --------
    pd.DataFrame
        Processed BIDS events dataframe
    """
    print(f"  Processing {len(df_run)} trials for run {run_number}")
    
    all_events = []
    
    # Process each trial to create multiple event types
    for idx, row in df_run.iterrows():
        trial_num = row['trial_number']
        
        # 1. Heat stimulus event
        heat_event = {
            'onset': round(row['stim_start_time'], 3),
            'duration': round(row['stim_end_time'] - row['stim_start_time'], 3),
            'trial_type': temp_mapping.get(row['stimulus_temp'], 
                                         f'temp{row["stimulus_temp"]:.1f}'.replace('.', 'p')),
            'temp_celsius': row['stimulus_temp'],
            'pain_binary': row['pain_binary_coded'],
            'vas_0_200': row['vas_final_coded_rating'],
            'block': run_number,
            'trial_index': trial_num,
            'stim_start_time': row['stim_start_time'],
            'stim_end_time': row['stim_end_time'],
            'vas_start_time': row['vas_start_time'],
            'vas_end_time': row['vas_end_time']
        }
        all_events.append(heat_event)
        
        # 2. Decision event (pain Yes/No question)
        decision_event = {
            'onset': round(row['pain_q_start_time'], 3),
            'duration': round(row['pain_q_end_time'] - row['pain_q_start_time'], 3),
            'trial_type': 'decision',
            'temp_celsius': row['stimulus_temp'],
            'pain_binary': row['pain_binary_coded'],
            'vas_0_200': row['vas_final_coded_rating'],
            'block': run_number,
            'trial_index': trial_num,
            'stim_start_time': row['stim_start_time'],
            'stim_end_time': row['stim_end_time'],
            'vas_start_time': row['vas_start_time'],
            'vas_end_time': row['vas_end_time']
        }
        all_events.append(decision_event)
        
        # 3. Rating event (VAS)
        rating_event = {
            'onset': round(row['vas_start_time'], 3),
            'duration': round(row['vas_end_time'] - row['vas_start_time'], 3),
            'trial_type': 'rating',
            'temp_celsius': row['stimulus_temp'],
            'pain_binary': row['pain_binary_coded'],
            'vas_0_200': row['vas_final_coded_rating'],
            'block': run_number,
            'trial_index': trial_num,
            'stim_start_time': row['stim_start_time'],
            'stim_end_time': row['stim_end_time'],
            'vas_start_time': row['vas_start_time'],
            'vas_end_time': row['vas_end_time']
        }
        all_events.append(rating_event)
        
        # 4. Optional delay event (between stimulus end and decision start)
        if include_delay:
            delay_duration = row['pain_q_start_time'] - row['stim_end_time']
            if delay_duration > 0:  # Only add if there's actually a delay
                delay_event = {
                    'onset': round(row['stim_end_time'], 3),
                    'duration': round(delay_duration, 3),
                    'trial_type': 'delay',
                    'temp_celsius': row['stimulus_temp'],
                    'pain_binary': row['pain_binary_coded'],
                    'vas_0_200': row['vas_final_coded_rating'],
                    'block': run_number,
                    'trial_index': trial_num,
                    'stim_start_time': row['stim_start_time'],
                    'stim_end_time': row['stim_end_time'],
                    'vas_start_time': row['vas_start_time'],
                    'vas_end_time': row['vas_end_time']
                }
                all_events.append(delay_event)
    
    # Convert to DataFrame
    events_out = pd.DataFrame(all_events)
    
    # Sort by onset time
    events_out = events_out.sort_values('onset').reset_index(drop=True)
    
    # Validate data
    event_counts = events_out['trial_type'].value_counts()
    print(f"    Event counts: {dict(event_counts)}")
    print(f"    Total events: {len(events_out)}")
    print(f"    Duration ranges by event type:")
    for event_type in events_out['trial_type'].unique():
        durations = events_out[events_out['trial_type'] == event_type]['duration']
        print(f"      {event_type}: {durations.min():.3f} - {durations.max():.3f} seconds")
    
    return events_out

def split_events_to_runs(input_file, output_dir, dry_run=False, include_delay=True, drop_log_path=None):
    """
    Split master events file into run-specific BIDS events files.
    
    Parameters:
    -----------
    input_file : str or Path
        Path to master events TSV file
    output_dir : str or Path  
        Output directory (should be subject's func/ folder)
    dry_run : bool
        If True, don't write files, just show what would be done
    include_delay : bool
        If True, include delay nuisance regressor events
    drop_log_path : str or Path, optional
        Path to EEG dropped_trials.tsv file
    """
    
    print(f"Loading master events file: {input_file}")
    
    # Load master events file
    try:
        df = pd.read_csv(input_file, sep='\t')
    except Exception as e:
        print(f"Error loading file: {e}")
        return False
        
    print(f"Loaded {len(df)} events across {df['run'].nunique()} runs")
    print(f"Runs found: {sorted(df['run'].unique())}")
    
    # Load EEG drop log if provided
    drop_log = pd.DataFrame()
    if drop_log_path:
        print(f"\nLoading EEG drop log: {drop_log_path}")
        drop_log = load_eeg_drop_log(drop_log_path)
        if not drop_log.empty:
            print(f"  Found {len(drop_log)} dropped trial(s)")
            # Filter out dropped trials
            n_before = len(df)
            drop_pairs = set(
                (int(row['run']), int(row['trial_number']))
                for _, row in drop_log.iterrows()
            )
            df = df[~df.apply(lambda r: (int(r['run']), int(r['trial_number'])) in drop_pairs, axis=1)]
            n_after = len(df)
            print(f"  Filtered out {n_before - n_after} trial(s)")
        else:
            print(f"  No EEG drop log found or invalid format")
    else:
        print("\nNo EEG drop log specified - processing all trials")
    
    # Setup temperature mapping
    temp_mapping = setup_temperature_mapping()
    
    # Create output directory
    output_path = Path(output_dir)
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
        
    print(f"\nOutput directory: {output_path}")
    
    # Process each run
    for run_num in sorted(df['run'].unique()):
        print(f"\n--- Processing Run {run_num} ---")
        
        # Filter data for this run
        df_run = df[df['run'] == run_num].copy()
        
        if len(df_run) == 0:
            print(f"  No events found for run {run_num}")
            continue
            
        # Process events for this run
        events_processed = process_run_events(df_run, run_num, temp_mapping, include_delay)
        
        # Generate output filename
        output_filename = f"sub-0001_task-pain_run-{run_num:02d}_events.tsv"
        output_file = output_path / output_filename
        
        print(f"  Output file: {output_filename}")
        
        if dry_run:
            print("  [DRY RUN] Would save file with columns:")
            print(f"    {list(events_processed.columns)}")
            print(f"  Sample rows:")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(events_processed.head(3).to_string(index=False))
        else:
            # Save to TSV
            events_processed.to_csv(output_file, sep='\t', index=False, float_format='%.6f')
            print(f"  ✓ Saved {len(events_processed)} events to {output_filename}")
            
    print(f"\n{'=== DRY RUN COMPLETED ===' if dry_run else '=== PROCESSING COMPLETED ==='}")
    
    return True

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Split master fMRI events file into run-specific BIDS events files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Process with default paths (includes delay events: 44 rows per run)
  python split_events_to_runs.py
  
  # Exclude delay events (33 rows per run: 11 heat + 11 decision + 11 rating)
  python split_events_to_runs.py --no_delay
  
  # Specify custom input and output
  python split_events_to_runs.py --input /path/to/events.tsv --output_dir /path/to/func/
  
  # Dry run to preview changes
  python split_events_to_runs.py --dry_run
        """
    )
    
    # Default paths
    default_input = r"C:\Users\joshu\EEG_fMRI_Analysis\fmri_pipeline\BIDS\dataset\sub-0001\func\sub-0001_task-thermalactive_events.tsv"
    default_output = r"C:\Users\joshu\EEG_fMRI_Analysis\fmri_pipeline\BIDS\dataset\sub-0001\func"
    
    parser.add_argument('--input', '-i', 
                       default=default_input,
                       help=f'Input master events TSV file (default: {default_input})')
    
    parser.add_argument('--output_dir', '-o',
                       default=default_output, 
                       help=f'Output directory for run-specific events files (default: {default_output})')
    
    parser.add_argument('--dry_run', '-d', action='store_true',
                       help='Dry run - show what would be done without writing files')
    
    parser.add_argument('--no_delay', action='store_true',
                       help='Exclude delay nuisance regressor events (default: include delay events)')
    
    parser.add_argument('--drop_log', '-e',
                       default=r"C:\Users\joshu\EEG_fMRI_Analysis\eeg_pipeline\bids_output\derivatives\sub-0001\eeg\features\dropped_trials.tsv",
                       help='Path to EEG dropped_trials.tsv file (default: auto-detect for sub-0001)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
        
    # Run the processing
    include_delay = not args.no_delay  # Default is to include delay
    success = split_events_to_runs(args.input, args.output_dir, args.dry_run, include_delay, args.drop_log)
    
    if success:
        if not args.dry_run:
            print("\n✓ Successfully created run-specific events files")
            print("\nFiles are ready for fMRIPrep/FSL/SPM GLM analysis!")
        return 0
    else:
        print("\n✗ Processing failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
