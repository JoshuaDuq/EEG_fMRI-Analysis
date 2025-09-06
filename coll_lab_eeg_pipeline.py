import importlib
import os

import mne
import pandas as pd
import pyprep
from mne_bids import BIDSPath, get_entities_from_fname
from mne_icalabel import label_components

os.environ["PYTHONUTF8"] = "1"
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
import warnings

import bct
import matplotlib
import numpy as np
import scipy
from joblib import Parallel, delayed
from mne.beamformer import apply_lcmv_epochs, make_lcmv
from mne.datasets import fetch_fsaverage
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_connectivity import (
    envelope_correlation,
    spectral_connectivity_epochs,
)
from specparam import SpectralModel

matplotlib.use("Agg")  # Use non-interactive backend to avoid tkinter errors
import subprocess
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from mne.time_frequency import AverageTFR
from nilearn.plotting import plot_connectome, plot_markers


def reject_bad_segs(raw, annot_to_reject="", delete_non_bads=True):
    """This function rejects all time spans annotated as annot_to_reject and concatenates the rest"""
    # this implementation seemed buggy, modified it here
    raw_segs = []
    if delete_non_bads:
        # Find idx of annotations not in annot_to_reject
        del_idx = [
            i for i, x in enumerate(raw.annotations.description) if x != annot_to_reject
        ]
        raw.annotations.delete(del_idx)  # delete the good annotations

    # Add the first segment before the first bad annotation if recording does not start with a bad annotation
    if len(raw.annotations.onset) > 0 and raw.annotations.onset[0] != 0:
        tmin = 0
        tmax = raw.annotations.onset[0]  # start at beginning of raw
        raw_segs.append(
            raw.copy().crop(  # this retains raw between tmin and tmax
                tmin=tmin,
                tmax=tmax,
                include_tmax=False,  # this is onset of first bad annot
            )
        )

    for jsegment in range(1, len(raw.annotations)):
        # print(raw.annotations.description[jsegment], annot_to_reject)
        if (
            raw.annotations.description[jsegment] == annot_to_reject
        ):  # Append all other than 'bad_ITI'
            tmin = (
                raw.annotations.onset[jsegment - 1]
                + raw.annotations.duration[jsegment - 1]
            )  # start at ending of last bad annot
            tmax = raw.annotations.onset[jsegment]  # end at onset of current bad annot
            raw_segs.append(
                raw.copy().crop(  # this retains raw between tmin and tmax
                    tmin=tmin,
                    tmax=tmax,
                    include_tmax=False,  # this is onset of bad annot
                )
            )

    # Add the last segment after the last bad annotation if recording does not end with a bad annotation
    if (
        len(raw.annotations.onset) > 0
        and raw.annotations.onset[-1] + raw.annotations.duration[-1] < raw.times[-1]
    ):
        tmin = (
            raw.annotations.onset[-1] + raw.annotations.duration[-1]
        )  # start at ending of last bad annot
        tmax = raw.times[-1]
        raw_segs.append(
            raw.copy().crop(  # this retains raw between tmin and tmax
                tmin=tmin,
                tmax=tmax,
                include_tmax=True,  # this is end of last bad annot
            )
        )

    return mne.concatenate_raws(raw_segs), len(raw_segs)


def run_bads_detection(
    bids_path,
    pipeline_path,
    task,
    session=None,
    ransac=False,
    repeats=3,
    average_reref=False,
    file_extension=".vhdr",
    montage="easycap-M1",
    delete_breaks=False,
    rename_anot_dict=None,
    breaks_min_length=20,
    t_start_after_previous=2,
    t_stop_before_next=2,
    overwrite_chans_tsv=True,
    consider_previous_bads=False,
    n_jobs=1,
    l_pass=100,
    notch=None,
    subjects="all",
    custom_bad_dict=None,
):
    """
    Run pyprep to detect bad channels and update the corresponding channels.tsv file.
    bids_path : str
        Path to the BIDS dataset.
    task : str
        The task to process.
    eog_chans : list
        List of EOG channels.
    misc_chans : list
        List of misc channels.
    ransac : bool
        Whether to use RANSAC to detect bad channels.
    repeats : int
        Number of times to repeat the bad channel detection.
    average_reref : bool
        Whether to re-reference to average.
    extension : str
        The extension of the EEG files.
    montage : str
        The montage to use.
    annot_breaks : bool
        Whether to annotate breaks.
    rename_anot_dict : dict or None
        Dictionary to rename annotations.
    overwrite_chans_tsv : bool
        Whether to overwrite the channels.tsv file. If False, a new file with the bad channels will be created.
    clear_previous_bads : bool
        Whether to clear previously marked bad channels.
    n_jobs : int
        Number of jobs to use. If 1, the process will be sequential.
    l_pass : float
        Low pass filter frequency to apply before detection. If None, no filtering will be applied.
        Note that pyprep applies a high pass filter at 1 Hz by default.
    notch: float
        Notch filter frequency to apply before detection. If None, no filtering will be applied.
    custom_bad_dict:
        Dictionary to identify bad channels. Keys are participant IDs and values are lists of bad channels selected. These will be added to the bad channels found by pyprep.
    """
    # Find the corresponding eeg file
    eeg_files = list(
        set(
            str(f)
            for f in BIDSPath(
                root=bids_path,
                task=task,
                session=session,
                datatype="eeg",
                suffix="eeg",
                extension=file_extension,
            ).match()
        )
    )
    eeg_files.sort()
    # For some datasets, same file is in sourcedata, root or derivatives because dataset is a modified BIDS
    # Remove sourcedata and derivatives files
    eeg_files = [
        f for f in eeg_files if "sourcedata" not in f or "derivatives" not in f
    ]
    if subjects != "all":
        # If subjects is not 'all', filter the eeg_files list
        eeg_files = [
            f
            for f in eeg_files
            if get_entities_from_fname(f).get("subject") in subjects
        ]

    logger.title(f"Custom step - Find bad channels in {len(eeg_files)} files.")

    # Initialize a data frame with file name and participants to store the results

    def _run_bads_detection(
        file,
        ransac=ransac,
        repeats=repeats,
        average_reref=average_reref,
        montage=montage,
        delete_breaks=delete_breaks,
        rename_anot_dict=rename_anot_dict,
        overwrite_chans_tsv=overwrite_chans_tsv,
        breaks_min_length=breaks_min_length,
        t_start_after_previous=t_start_after_previous,
        t_stop_before_next=t_stop_before_next,
        consider_previous_bads=consider_previous_bads,
        l_pass=l_pass,
        notch=notch,
        custom_bad_dict=custom_bad_dict,
    ):
        from mne_bids_pipeline._logging import logger

        # Initialize the bads_frame
        bads_frame = pd.DataFrame(
            data=None,
            columns=[
                "file_name",
                "participant_id",
                "session",
                "n_bads",
                "bad_channels",
                "n_breaks_found",
                "recording_duration",
                "success",
                "error",
            ],
        )
        # Add filename to bads_frame using only the file name remove rest of path
        bads_frame.loc[file, "file_name"] = os.path.basename(file)

        # Try to find bad channels using pypred. Catch errors and add to bads_frame
        try:
            with mne.utils.use_log_level(False):
                msg = "Finding bad channels using pyprep."
                # Get subject from file
                logger.info(
                    **gen_log_kwargs(
                        message=msg,
                        subject=get_entities_from_fname(file)["subject"],
                        session=get_entities_from_fname(file)["session"],
                    )
                )
                # Load corresponding chan file
                chan_file = pd.read_csv(
                    file.replace("_eeg.vhdr", "_channels.tsv"), sep="\t"
                )

                # Add participant_id and session to bads_frame
                bads_frame.loc[file, "participant_id"] = get_entities_from_fname(file)[
                    "subject"
                ]
                if get_entities_from_fname(file).get("session") is not None:
                    bads_frame.loc[file, "session"] = get_entities_from_fname(file)[
                        "session"
                    ]

                previous_bads = chan_file[
                    (chan_file["status"] == "bad")
                    & (chan_file["type"].isin(["eeg", "EEG"]))
                ]["name"].tolist()
                # Get eog, ecg, emg and misc channels if type column is EEG, EOG, EMG, ECG
                if previous_bads:
                    # Log how many bad channels were found
                    if not consider_previous_bads:
                        msg = f"Found {len(previous_bads)} bad channels already marked. THOSE WILL BE IGNORED AND CLEARED BECAUSE consider_previous_bads=False."
                    else:
                        msg = f"Found {len(previous_bads)} bad channels already marked. THOSE WILL BE CONSIDERED BECAUSE consider_previous_bads=True."
                    logger.info(
                        **gen_log_kwargs(
                            message=msg,
                            subject=get_entities_from_fname(file)["subject"],
                            session=get_entities_from_fname(file)["session"],
                            emoji="⚠️",
                        )
                    )

                eog_chans = chan_file.loc[
                    chan_file["type"].isin(["EOG", "eog"]), "name"
                ].tolist()
                ecg_chans = chan_file.loc[
                    chan_file["type"].isin(["ecg", "ECG"]), "name"
                ].tolist()
                emg_chans = chan_file.loc[
                    chan_file["type"].isin(["EMG", "emg"]), "name"
                ].tolist()
                misc_chans = chan_file.loc[
                    chan_file["type"].isin(["MISC", "misc"]), "name"
                ].tolist()

                # For painlaval quick fix, remove later. IF GSR in type, replace with misc
                if "GSR" in chan_file["name"].tolist():
                    chan_file.loc[chan_file["type"] == "GSR", "type"] = "MISC"
                    misc_chans = misc_chans + ["GSR"]

                # read input data
                raw = mne.io.read_raw(
                    file,
                    preload=True,
                    verbose=False,
                    eog=eog_chans,
                    misc=misc_chans + ecg_chans + emg_chans,
                )
                assert isinstance(raw, mne.io.BaseRaw)

                raw.set_montage(montage)

                if l_pass:
                    raw.filter(None, l_pass)
                if notch:
                    raw.notch_filter(notch, picks="eeg", verbose=False)

                # Set previous bads
                if consider_previous_bads:
                    raw.info["bads"] = list(set(raw.info["bads"] + previous_bads))

                # Annotate breaks
                if delete_breaks:
                    annot_breaks = mne.preprocessing.annotate_break(
                        raw=raw,
                        min_break_duration=breaks_min_length,
                        t_start_after_previous=t_start_after_previous,
                        t_stop_before_next=t_stop_before_next,
                        ignore=(
                            "bad",
                            "edge",
                            "New Segment",
                        ),
                    )
                    raw.set_annotations(raw.annotations + annot_breaks)

                    original_dur = raw.times[-1]

                    # We remove the breaks from the data because pyprep
                    # is not yet able to handle "BAD_break" annotations
                    # This creates discontinuities in the data but should
                    # not affect the bad channel detection too much
                    raw, n_blocks = reject_bad_segs(raw, annot_to_reject="BAD_break")

                    new_dur = raw.times[-1]
                    msg = f"Found {len(annot_breaks)} breaks and in the data and {n_blocks} valid segments."
                    logger.info(
                        **gen_log_kwargs(
                            message=msg,
                            subject=get_entities_from_fname(file)["subject"],
                            session=get_entities_from_fname(file)["session"],
                            emoji="⚠️",
                        )
                    )
                    msg = f"Removed {original_dur - new_dur} s of breaks from the data."
                    logger.info(
                        **gen_log_kwargs(
                            message=msg,
                            subject=get_entities_from_fname(file)["subject"],
                            session=get_entities_from_fname(file)["session"],
                            emoji="⚠️",
                        )
                    )
                    bads_frame.loc[file, "n_breaks_found"] = len(annot_breaks)
                    bads_frame.loc[file, "recording_duration"] = original_dur - new_dur

                # Rename bad boundary annotations
                if rename_anot_dict:
                    raw.annotations.rename(rename_anot_dict)

                # In some datasets referenced to FCz, channels around the ref are
                # considered flat by pyprep, so we migh want to re-reference to average
                if average_reref:
                    raw.set_eeg_reference("average")

                all_bads: list[str] = []

                for _ in range(repeats):
                    # Find noisy channels, already detrended
                    nc = pyprep.NoisyChannels(raw=raw, random_state=42)
                    nc.find_bad_by_deviation()
                    nc.find_bad_by_correlation()
                    if ransac:
                        nc.find_bad_by_ransac()
                    bads = nc.get_bads()
                    all_bads.extend(bads)

                # Deduplicate across repeats and update raw.info
                all_bads = sorted(set(all_bads))
                raw.info["bads"] = all_bads

                # Add custom bad channels if provided
                if custom_bad_dict is not None:
                    task = get_entities_from_fname(file)["task"]
                    sub = get_entities_from_fname(file)["subject"]
                    # Check if task is in custom_bad_dict:
                    if task in custom_bad_dict:
                        if sub in custom_bad_dict[task]:
                            all_bads.extend(custom_bad_dict[task][sub])
                            all_bads = sorted(set(all_bads))
                            # Log how many custom bad channels were found
                            msg = f"Found {len(custom_bad_dict[task][sub])} custom bad channels: {custom_bad_dict[task][sub]}."
                            logger.info(
                                **gen_log_kwargs(
                                    message=msg,
                                    subject=get_entities_from_fname(file)["subject"],
                                    session=get_entities_from_fname(file)["session"],
                                    emoji="⚠️",
                                )
                            )
                            removed_custom_bads = [
                                ch
                                for ch in custom_bad_dict[task][sub]
                                if ch not in raw.info["bads"]
                            ]
                        else:
                            removed_custom_bads = []
                    else:
                        removed_custom_bads = []
                else:
                    removed_custom_bads = []

                # Log how many bad channels were found (use unique list)
                bad_chans = sorted(set(all_bads))

                # Set type of column description to string
                if "description" in chan_file.columns:
                    chan_file["description"] = chan_file["description"].astype(str)

                if not consider_previous_bads:
                    # Set all EEG channels to good
                    chan_file.loc[chan_file["type"].isin(["EEG", "eeg"]), "status"] = (
                        "good"
                    )
                    chan_file.loc[
                        chan_file["type"].isin(["EEG", "eeg"]), "description"
                    ] = ""
                # Flag bad channels
                for ch in bad_chans:
                    chan_file.loc[chan_file["name"] == ch, "status"] = "bad"
                    chan_file.loc[chan_file["name"] == ch, "description"] = (
                        "Bad channel detected by pyprep"
                    )
                    # Different description if ch in custom bad-channel list for this task/subject
                    if custom_bad_dict is not None and ch in custom_bad_dict.get(task, {}).get(sub, []):
                        chan_file.loc[chan_file["name"] == ch, "description"] = (
                            "Bad channel from custom bad channel list"
                        )

                # Save the file
                if overwrite_chans_tsv:
                    chan_file.to_csv(
                        file.replace("_eeg.vhdr", "_channels.tsv"),
                        sep="\t",
                        index=False,
                    )
                else:
                    chan_file.to_csv(
                        file.replace("_eeg.vhdr", "_channels.tsv").replace(
                            ".tsv", "_bad_channels.tsv"
                        ),
                        sep="\t",
                        index=False,
                    )
                msg = f"Found {len(raw.info['bads'])} bad channels using pyprep: {raw.info['bads']} and {len(removed_custom_bads)} custom bad channels that were not detected by pyprep: {removed_custom_bads} for a total of {len(all_bads)} bad channels."
                bads_frame.loc[file, "n_bads"] = len(all_bads)
                bads_frame.loc[file, "bad_channels"] = bad_chans

                # Get subject from file
                logger.info(
                    **gen_log_kwargs(
                        message=msg,
                        subject=get_entities_from_fname(file)["subject"],
                        session=get_entities_from_fname(file)["session"],
                        emoji="✅",
                    )
                )
        except Exception as e:
            bads_frame.loc[file, "success"] = 0
            bads_frame.loc[file, "error"] = str(e)
            # Log the error with a danger emoji
            logger.error(
                **gen_log_kwargs(
                    message=f"Error while finding bad channels in {file}: {e}",
                    subject=get_entities_from_fname(file)["subject"],
                    session=get_entities_from_fname(file)["session"],
                    emoji="❌",
                )
            )
        # Add all the functions input parameters to the bads_frame
        bads_frame.loc[file, "ransac"] = ransac
        bads_frame.loc[file, "repeats"] = repeats
        bads_frame.loc[file, "average_reref"] = average_reref
        bads_frame.loc[file, "montage"] = montage
        bads_frame.loc[file, "delete_breaks"] = delete_breaks
        bads_frame.loc[file, "rename_anot_dict"] = str(rename_anot_dict)
        bads_frame.loc[file, "overwrite_chans_tsv"] = overwrite_chans_tsv
        bads_frame.loc[file, "breaks_min_length"] = breaks_min_length
        bads_frame.loc[file, "t_start_after_previous"] = t_start_after_previous
        bads_frame.loc[file, "t_stop_before_next"] = t_stop_before_next
        bads_frame.loc[file, "consider_previous_bads"] = consider_previous_bads
        bads_frame.loc[file, "l_pass"] = l_pass
        bads_frame.loc[file, "notch"] = notch
        if custom_bad_dict is not None:
            bads_frame.loc[file, "custom_bad_dict"] = str(custom_bad_dict)
        else:
            bads_frame.loc[file, "custom_bad_dict"] = "None"

        # Return the bads_frame
        bads_frame.loc[file, "success"] = 1
        bads_frame.loc[file, "error_log"] = ""

        return bads_frame

    # Use joblib to parallelize the process
    if n_jobs != 1:
        bads_frame_list = Parallel(n_jobs=n_jobs)(
            delayed(_run_bads_detection)(file) for file in eeg_files
        )
    else:
        bads_frame_list = []
        for file in eeg_files:
            bframe = _run_bads_detection(file)
            bads_frame_list.append(bframe)
    if len(bads_frame_list) > 1:
        # Concatenate the bad_ica_frames
        bads_frame = pd.concat(bads_frame_list, ignore_index=False)
    else:
        # If there is only one bad_ica_frame, just use it
        bads_frame = bads_frame_list[0]
    # Save the bads_frame to a file in the root derivatives folder
    if not os.path.exists(pipeline_path):
        os.makedirs(pipeline_path)
    bads_frame.to_csv(
        os.path.join(pipeline_path, f"pyprep_task_{task}_log.csv"), index=False
    )


def run_ica_label(
    pipeline_path,
    task,
    prob_threshold=0.8,
    labels_to_keep=["brain", "other"],
    n_jobs=1,
    keep_mnebids_bads=False,
    subjects="all",
):
    """
    Run ICA label and flag bad components.
    pipeline_path : str
        Path to the pipeline.
    p : str
        The participant.
    task : str
        The task to process.
    prob_threshold : float
        The probability threshold to flag bad components.
    labels_to_keep : list
        The labels to keep. If a component is labeled as one of these labels, it will not be flagged as bad.
    n_jobs : int
        Number of jobs to use. If 1, the process will be sequential.
    keep_mnebids_bads : bool
        Whether to keep the bad components flagged by mne-bids pipeline. If False, the status will be set to good and the description will be empty if
        not flagged as bad by mne_icalabel.
    """
    ica_files = list(
        set(
            str(f)
            for f in BIDSPath(
                root=pipeline_path,
                task=task,
                session=None,
                suffix="ica",
                processing="icafit",
                extension=".fif",
                check=False,
            ).match()
        )
    )

    if subjects != "all":
        # If subjects is not 'all', filter the ica_files list
        ica_files = [
            f
            for f in ica_files
            if get_entities_from_fname(f).get("subject") in subjects
        ]

    ica_files.sort()

    logger.title(
        "Custom step - Find bad ICs using mne_icalabel in %d files." % len(ica_files)
    )

    def _run_ica_label(
        p,
        prob_threshold=prob_threshold,
        labels_to_keep=labels_to_keep,
        keep_mnebids_bads=keep_mnebids_bads,
    ):
        from mne_bids_pipeline._logging import logger

        # Create a bad_ica dataframe to store the results
        bad_ica_frame = pd.DataFrame(
            index=None,
            columns=[
                "file_name",
                "participant_id",
                "session",
                "n_bad_icas",
                "bad_icas",
            ],
        )

        # Check if session is in the path
        if "ses-" in p:
            ses_num = get_entities_from_fname(p)["session"]
        else:
            ses_num = None
        sub_num = get_entities_from_fname(p)["subject"]
        # Add file name, participant_id and session to the bad_ica_frame
        bad_ica_frame.loc[p, "file_name"] = os.path.basename(p)
        bad_ica_frame.loc[p, "participant_id"] = sub_num
        bad_ica_frame.loc[p, "session"] = ses_num

        with mne.utils.use_log_level(False):
            # Load ica file
            msg = "Finding bad icas using mne-icalabel."
            # Get subject from file
            logger.info(**gen_log_kwargs(message=msg, subject=sub_num, session=ses_num))

            ica = mne.preprocessing.read_ica(p)
            ica_epo = mne.read_epochs(
                p.replace("_proc-icafit_ica.fif", "_proc-icafit_epo.fif")
            )

            # Set average reference (required for iclabel)
            ica_epo.set_eeg_reference("average")

            # Get ICA labels and probabilities
            icalabel = label_components(ica_epo, ica, method="iclabel")
            icalabel["labels"] = np.asanyarray(icalabel["labels"])

            # Flag bad components with probability > 0.7 and labels that are not in labels_to_keep
            bad_comps = np.where(
                (icalabel["y_pred_proba"] > prob_threshold)
                & (~np.isin(icalabel["labels"], labels_to_keep))
            )[0]

            # Log how many components were flagged
            msg = f"Found {len(bad_comps)} bad components."
            logger.info(
                **gen_log_kwargs(
                    message=msg, subject=sub_num, session=ses_num, emoji="✅"
                )
            )
            # Add the number of bad components to the bad_ica_frame
            bad_ica_frame.loc[p, "n_bad_icas"] = len(bad_comps)
            bad_ica_frame.loc[p, "bad_icas"] = ", ".join([str(c) for c in bad_comps])

            # Load corresponding ica file
            try:
                # Check if the components.tsv file exists or create an empty one
                if os.path.exists(
                    p.replace("_proc-icafit_ica.fif", "_proc-ica_components.tsv")
                ):
                    ica_frame = pd.read_csv(
                        p.replace("_proc-icafit_ica.fif", "_proc-ica_components.tsv"),
                        sep="\t",
                    )
                else:
                    n_components = ica.n_components_
                    ica_frame = pd.DataFrame(
                        columns=[
                            "component",
                            "type",
                            "status",
                            "status_description",
                            "mne_icalabel_labels",
                            "mne_icalabel_proba",
                        ]
                    )
                    ica_frame["component"] = np.arange(n_components)

                # Reset the status (the EOG/ECG detection in the pipeline is not playing well with some datasets)
                if not keep_mnebids_bads:
                    ica_frame["status"] = "good"
                    ica_frame["status_description"] = ""

                # Flag bad components with columns status as bad and description as "Bad component detected by mne_icalabel"
                for comp in bad_comps:
                    ica_frame.loc[ica_frame["component"] == comp, "status"] = "bad"
                    ica_frame.loc[
                        ica_frame["component"] == comp, "status_description"
                    ] = "Bad component detected by mne_icalabel"

                # Add the labels and probabilities to the ica_frame
                ica_frame["mne_icalabel_labels"] = icalabel["labels"]
                ica_frame["mne_icalabel_proba"] = icalabel["y_pred_proba"]

                # Save the file
                ica_frame.to_csv(
                    p.replace("_proc-icafit_ica.fif", "_proc-ica_components.tsv"),
                    sep="\t",
                    index=False,
                )
                bad_ica_frame.loc[p, "success"] = 1
                bad_ica_frame.loc[p, "error_log"] = ""
                # Save the ica file with the new bad components flagged
                ica.exclude = bad_comps.tolist()
                ica.save(
                    p.replace("_proc-icafit_ica.fif", "_proc-ica_ica.fif"),
                    overwrite=True,
                )

            except Exception as e:
                # Log the error with a danger emoji
                logger.error(
                    **gen_log_kwargs(
                        message=f"Error while finding bad components in {p}: {e}, skipping this file",
                        subject=sub_num,
                        session=ses_num,
                        emoji="❌",
                    )
                )
                bad_ica_frame.loc[p, "success"] = 0
                bad_ica_frame.loc[p, "error_log"] = str(e)
        return bad_ica_frame

    # Use joblib to parallelize the process
    if n_jobs != 1:
        bad_ica_frames = Parallel(n_jobs=n_jobs)(
            delayed(_run_ica_label)(p) for p in ica_files
        )
    else:
        bad_ica_frames = []
        for p in ica_files:
            _run_ica_label(p)
            bad_ica_frames.append(_run_ica_label(p))

    if len(bad_ica_frames) > 1:
        # Concatenate the bad_ica_frames
        bad_ica_frame = pd.concat(bad_ica_frames, ignore_index=False)
    else:
        # If there is only one bad_ica_frame, just use it
        bad_ica_frame = bad_ica_frames[0]
    # Save the bad_ica_frame to a file in the root derivatives folder
    bad_ica_frame.to_csv(
        os.path.join(pipeline_path, f"icalabel_task_{task}_log.csv"),
        sep="\t",
        index=False,
    )


def update_config(config, new_values, outfile=None):
    # Read .py file
    with open(config) as file:
        lines = file.readlines()

    # Find the lines to replace
    for key, value in new_values.items():
        found = False  # Initialize found for each key
        for i, line in enumerate(lines):
            if line.replace(" ", "").startswith(key + "=") and "no update" not in line:
                # Replace the line
                if isinstance(value, str):
                    lines[i] = f"{key} = '{value}'\n"
                else:
                    lines[i] = f"{key} = {value}\n"
                found = True
        if not found:
            # If the key was not found, create a new line
            lines.append("\n")
            lines.append(f"{key} = {value}\n")

    if outfile is not None:
        # Write the new file
        with open(outfile, "w") as file:
            file.writelines(lines)
    else:
        # Overwrite the original file
        with open(config, "w") as file:
            file.writelines(lines)


def get_specific_config(config_file, prefix):
    spec = importlib.util.spec_from_file_location(
        name="custom_config", location=config_file
    )

    custom_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_cfg)
    config = {}
    for key in dir(custom_cfg):
        if prefix + "_" in key:
            val = getattr(custom_cfg, key)
            config[key.replace(prefix + "_", "")] = val

    return config


def get_features_config(config_file):
    spec = importlib.util.spec_from_file_location(
        name="custom_config", location=config_file
    )

    custom_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_cfg)
    features_config = {}
    for key in dir(custom_cfg):
        if "features_" in key:
            val = getattr(custom_cfg, key)
            features_config[key] = val

    return features_config


def get_config_keyval(config_file, key, return_false_if_not_found=True):
    spec = importlib.util.spec_from_file_location(
        name="custom_config", location=config_file
    )

    custom_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_cfg)
    # Find the value of the key
    for k in dir(custom_cfg):
        if key == k:  # Use exact match
            val = getattr(custom_cfg, k)
            return val
    if return_false_if_not_found:
        # If the key is not found, return False
        return False
    else:
        raise ValueError(f"Key {key} not found in {config_file}")


def collect_preprocessing_stats(bids_path, pipeline_path, task):
    mne.set_log_level("ERROR")

    # Initialize the dataframe
    preprocessing_stats = pd.DataFrame(data=None, columns=["participant_id", "session"])
    preprocessing_stats.set_index(["participant_id", "session"], inplace=True)

    clean_epo_files = list(
        set(
            str(f)
            for f in BIDSPath(
                root=pipeline_path,
                task=task,
                session=None,
                suffix="epo",
                processing="clean",
                extension=".fif",
                check=False,
            ).match()
        )
    )

    # Number of bad channels
    for p in clean_epo_files:
        sess_num = get_entities_from_fname(p)["session"]
        # If there is no session, all sessions are 1
        if not sess_num:
            sess_num = "1"

        sub_num = "sub-" + get_entities_from_fname(p)["subject"]

        # Get the channel file - try consolidated first, then run-specific
        chan_filename = p.replace("_proc-clean_epo.fif", "_bads.tsv")
        
        if not os.path.exists(chan_filename):
            # Try to find run-specific bads files and aggregate them
            import glob
            run_pattern = chan_filename.replace("_bads.tsv", "_run-*_bads.tsv")
            run_bads_files = glob.glob(run_pattern)
            
            if run_bads_files:
                # Read all run-specific bads files and get unique bad channels
                all_bad_channels = set()
                for run_file in run_bads_files:
                    run_bads = pd.read_csv(run_file, sep="\t")
                    all_bad_channels.update(run_bads['name'].tolist())
                
                # Create a temporary dataframe with unique bad channels
                chan_file = pd.DataFrame({'name': list(all_bad_channels)})
            else:
                # No bads files found, assume no bad channels
                logger.warning(f"No bads file found for {sub_num}, assuming no bad channels")
                chan_file = pd.DataFrame({'name': []})
        else:
            chan_file = pd.read_csv(chan_filename, sep="\t")

        msg = "Collecting preprocessing stats."
        logger.info(
            **gen_log_kwargs(
                message=msg, subject=sub_num.replace("sub", ""), session=sess_num
            )
        )
        # Number of bad channels
        preprocessing_stats.loc[(sub_num, sess_num), "n_bad_channels"] = len(chan_file)

        # Number of removed ICA components
        ica_frame = pd.read_csv(
            p.replace("_proc-clean_epo.fif", "_proc-ica_components.tsv"),
            sep="\t",
        )

        # Number of bad icas
        preprocessing_stats.loc[(sub_num, sess_num), "n_bad_ica"] = len(
            ica_frame[ica_frame["status"] == "bad"]
        )

        # Number of total/removed epochs
        epochs = mne.read_epochs(p)
        # Get the events present in the epochs
        events_type = list(epochs.event_id.keys())
        preprocessing_stats.loc[(sub_num, sess_num), "total_clean_epochs"] = len(epochs)
        preprocessing_stats.loc[(sub_num, sess_num), "n_removed_epochs"] = len(
            epochs.drop_log
        ) - len(epochs)

        # Add number of epochs flagged because of boundary events
        n_boundary = len(
            [i for i, x in enumerate(epochs.drop_log) if "BAD boundary" in x]
        )
        preprocessing_stats.loc[(sub_num, sess_num), "boundary_n_removed_epochs"] = (
            n_boundary
        )

        # Get epochs removed and remaining in each event type
        for event in events_type:
            epochs_cond = epochs[event]
            preprocessing_stats.loc[
                (sub_num, sess_num), event + "_total_clean_epochs"
            ] = len(epochs_cond)

    preprocessing_stats.sort_values(by=["participant_id", "session"], inplace=True)
    preprocessing_stats.reset_index(inplace=True)
    preprocessing_stats.to_csv(
        os.path.join(pipeline_path, f"task_{task}_preprocessing_stats.tsv"),
        sep="\t",
        index=False,
    )
    preprocessing_stats.describe().to_csv(
        os.path.join(pipeline_path, f"task_{task}_preprocessing_stats_desc.tsv"),
        sep="\t",
        index=False,
    )


def validate_features_config(
    connectivity_methods,
    graph_metrics,
    connectivity_threshold,
    freq_bands,
    psd_freqmin,
    psd_freqmax,
    freq_res,
    compute_connectivity,
    compute_sourcespace_features,
):
    """
    Validate configuration parameters for feature extraction.

    Raises:
        ValueError: If any configuration parameter is invalid
    """
    errors = []

    # Validate connectivity methods
    valid_connectivity_methods = ["aec", "wpli"]
    if connectivity_methods:
        invalid_methods = [
            m for m in connectivity_methods if m not in valid_connectivity_methods
        ]
        if invalid_methods:
            errors.append(
                f"Invalid connectivity methods: {invalid_methods}. Valid methods are: {valid_connectivity_methods}"
            )

    # Validate graph metrics
    valid_graph_metrics = ["density", "clustering", "path_length", "efficiency"]
    if graph_metrics:
        invalid_metrics = [m for m in graph_metrics if m not in valid_graph_metrics]
        if invalid_metrics:
            errors.append(
                f"Invalid graph metrics: {invalid_metrics}. Valid metrics are: {valid_graph_metrics}"
            )

    # Validate connectivity threshold
    if not (0.0 <= connectivity_threshold <= 1.0):
        errors.append(
            f"Connectivity threshold must be between 0.0 and 1.0, got: {connectivity_threshold}"
        )

    # Validate frequency parameters
    if psd_freqmin >= psd_freqmax:
        errors.append(
            f"PSD minimum frequency ({psd_freqmin}) must be less than maximum frequency ({psd_freqmax})"
        )

    if freq_res <= 0:
        errors.append(f"Frequency resolution must be positive, got: {freq_res}")

    # Validate frequency bands
    if not freq_bands:
        errors.append("Frequency bands dictionary cannot be empty")
    else:
        for band_name, (low, high) in freq_bands.items():
            if low >= high:
                errors.append(
                    f"Invalid frequency band '{band_name}': low frequency ({low}) must be less than high frequency ({high})"
                )
            if low < 0:
                errors.append(
                    f"Invalid frequency band '{band_name}': frequencies must be non-negative, got low={low}"
                )

    # Validate connectivity configuration consistency
    if compute_connectivity and not compute_sourcespace_features:
        errors.append(
            "Connectivity analysis requires source space features to be enabled (compute_sourcespace_features=True)"
        )

    if compute_connectivity and not connectivity_methods:
        errors.append(
            "Connectivity analysis is enabled but no connectivity methods specified"
        )

    if connectivity_methods and not compute_connectivity:
        logger.warning(
            "Connectivity methods specified but connectivity analysis is disabled (compute_connectivity=False)"
        )

    if graph_metrics and not compute_connectivity:
        logger.warning(
            "Graph metrics specified but connectivity analysis is disabled (compute_connectivity=False)"
        )

    # Raise error if any validation failed
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        raise ValueError(error_msg)

    logger.info("✅ Configuration validation passed")


def compute_features(
    out_path,
    bids_path,
    task,
    freq_bands,
    sourcecoords_file,
    freq_res=1,
    somato_chans=None,
    psd_freqmax=100,
    psd_freqmin=1,
    n_jobs=1,
    subjects="all",
    interpolate_bads=True,
    compute_sourcespace_features=True,
    task_is_rest=True,
    roi_channels=None,
    compute_connectivity=True,
    connectivity_methods=["aec", "wpli"],
    graph_metrics=["density", "clustering", "path_length", "efficiency"],
    connectivity_threshold=0.8,
):
    logger.title("Custom step - Computing features")

    # Validate configuration parameters
    validate_features_config(
        connectivity_methods=connectivity_methods,
        graph_metrics=graph_metrics,
        connectivity_threshold=connectivity_threshold,
        freq_bands=freq_bands,
        psd_freqmin=psd_freqmin,
        psd_freqmax=psd_freqmax,
        freq_res=freq_res,
        compute_connectivity=compute_connectivity,
        compute_sourcespace_features=compute_sourcespace_features,
    )

    # Make sure the output path exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    clean_epo_files = list(
        set(
            str(f)
            for f in BIDSPath(
                root=bids_path,
                task=task,
                session=None,
                suffix="epo",
                processing="clean",
                extension=".fif",
                check=False,
            ).match()
        )
    )
    clean_epo_files.sort()

    # Get participants
    if subjects != "all":
        clean_epo_files = [
            f
            for f in clean_epo_files
            if get_entities_from_fname(f)["subject"] in subjects
        ]

    def _compute_features(
        p,
        task=task,
        out_path=out_path,
        freq_bands=freq_bands,
        freq_res=freq_res,
        somato_chans=somato_chans,
        psd_freqmax=psd_freqmax,
        psd_freqmin=psd_freqmin,
        sourcecoords_file=sourcecoords_file,
        interpolate_bads=interpolate_bads,
        compute_sourcespace_features=compute_sourcespace_features,
        task_is_rest=task_is_rest,
        roi_channels=roi_channels,
        compute_connectivity=compute_connectivity,
        connectivity_methods=connectivity_methods,
        graph_metrics=graph_metrics,
        connectivity_threshold=connectivity_threshold,
    ):
        sub_num = "sub-" + get_entities_from_fname(p)["subject"]
        session = get_entities_from_fname(p)["session"]
        # If there is no session, all sessions are 1 and do not create a session folder
        if not get_entities_from_fname(p)["session"]:
            session = "1"
            session_out = ""
            session_file = ""
        else:
            session_out = "ses-" + session
            session_file = "ses-" + session + "_"

        # Create subject folder with eeg subdirectory to match BIDS structure
        subject_eeg_dir = os.path.join(out_path, sub_num, session_out, "eeg")
        os.makedirs(subject_eeg_dir, exist_ok=True)

        # Load preprocessed epochs
        clean_epo = mne.read_epochs(p)
        if interpolate_bads:
            clean_epo = clean_epo.interpolate_bads()

        # Keep only EEG channels
        clean_epo = clean_epo.pick_types(eeg=True)

        # Initialize features frame based on task type
        if task_is_rest:
            # Per-subject features (original behavior)
            # Create initial row data with explicit data types
            features_frame = pd.DataFrame(
                {
                    "participant_id": [sub_num],
                    "session": [session],
                    "task": [task],
                    "file_name": [p],
                }
            )
            # Convert session to string to ensure consistency
            features_frame["session"] = features_frame["session"].astype(str)
            features_frame.set_index(["participant_id", "session"], inplace=True)
        else:
            # Per-trial features (new behavior)
            features_frame = pd.DataFrame(
                columns=[
                    "participant_id",
                    "session",
                    "trial_id",
                    "condition",
                    "task",
                    "file_name",
                ]
            )

        if task_is_rest:
            # Original rest task behavior - no condition loop
            # logger
            msg = "Computing features - PSD."
            logger.info(
                **gen_log_kwargs(
                    message=msg, subject=sub_num.replace("sub-", ""), session=session
                )
            )

            # Zero pad the data to required duration to get the desired frequency resolution
            pad_s = (
                1 / freq_res
            )  # Padding necessary to get the desired frequency resolution
            epo_dur = clean_epo.get_data().shape[2] / clean_epo.info["sfreq"]
            n_pad = int(int((pad_s - epo_dur) * clean_epo.info["sfreq"]) / 2)
            clean_epo_padded = mne.EpochsArray(
                np.pad(
                    clean_epo.get_data(),
                    pad_width=((0, 0), (0, 0), (n_pad, n_pad)),
                    mode="constant",
                    constant_values=0,
                ),
                info=clean_epo.info,
                tmin=-pad_s / 2,
                verbose=False,
            )

            # Compute power spectral density (and average across epochs)
            psd = clean_epo_padded.compute_psd(
                method="multitaper", n_jobs=-1, fmin=psd_freqmin, fmax=psd_freqmax
            ).average()
            # Save PSD
            psd.save(
                os.path.join(
                    subject_eeg_dir, f"{sub_num}_task-{task}_{session_file}psd.h5"
                ),
                overwrite=True,
            )

            # Source space analysis for rest task
            if compute_sourcespace_features and compute_connectivity:
                #############################################################
                # Source space
                #############################################################
                msg = "Computing features - source space for rest task"
                logger.info(
                    **gen_log_kwargs(
                        message=msg,
                        subject=sub_num.replace("sub-", ""),
                        session=session,
                    )
                )

                # For connectivity, use the rest epochs
                clean_epo_connectivity = clean_epo.copy().set_eeg_reference(
                    projection=True
                )
        else:
            # Task-based behavior - process each condition separately
            # We will iterate through the original event IDs and sanitize them inside the loop.
            original_event_id = clean_epo.event_id.copy()
            conditions_to_process = list(original_event_id.keys())

            msg = f"Computing features - PSD (task-based, {len(conditions_to_process)} conditions: {conditions_to_process})."
            logger.info(
                **gen_log_kwargs(
                    message=msg, subject=sub_num.replace("sub-", ""), session=session
                )
            )

            # Process each condition
            for condition in conditions_to_process:
                # 1. Subset epochs using the original (potentially slash-containing) condition name
                epochs_for_condition = clean_epo[condition].copy()

                # 2. Sanitize the name for this specific subset
                sanitized_name = condition.replace("/", "_").replace(" ", "")

                # 3. CRITICAL STEP: Replace the event_id dictionary on the *subsetted* epochs object
                #    This ensures that any downstream function only sees the clean name.
                epochs_for_condition.event_id = {
                    sanitized_name: original_event_id[condition]
                }

                logger.info(
                    **gen_log_kwargs(
                        message=f"Processing condition: {condition} -> {sanitized_name}",
                        subject=sub_num.replace("sub-", ""),
                        session=session,
                    )
                )

                condition_suffix = "_" + sanitized_name

                # Zero pad the data
                pad_s = 1 / freq_res
                epo_dur = (
                    epochs_for_condition.get_data().shape[2]
                    / epochs_for_condition.info["sfreq"]
                )

                if epo_dur < pad_s:
                    n_pad = int(
                        int((pad_s - epo_dur) * epochs_for_condition.info["sfreq"]) / 2
                    )
                    clean_epo_padded = mne.EpochsArray(
                        np.pad(
                            epochs_for_condition.get_data(),
                            pad_width=((0, 0), (0, 0), (n_pad, n_pad)),
                            mode="constant",
                            constant_values=0,
                        ),
                        info=epochs_for_condition.info,
                        tmin=-pad_s / 2,
                        event_id=epochs_for_condition.event_id,  # Pass the sanitized event_id
                        verbose=False,
                    )
                else:
                    clean_epo_padded = epochs_for_condition

                # Compute the PSD. It will inherit the sanitized event_id.
                psd = clean_epo_padded.compute_psd(
                    method="multitaper", n_jobs=-1, fmin=psd_freqmin, fmax=psd_freqmax
                )

                # This save operation will now succeed.
                psd.save(
                    os.path.join(
                        subject_eeg_dir,
                        f"{sub_num}_task-{task}_{session_file}psd{condition_suffix}.h5",
                    ),
                    overwrite=True,
                )

                # Peak alpha power and COG (global, each channel, somato-sensory, and ROIs)
                #############################################################
                msg = (
                    f"Computing features - Peak alpha for condition: {sanitized_name}."
                )
                logger.info(
                    **gen_log_kwargs(
                        message=msg,
                        subject=sub_num.replace("sub-", ""),
                        session=session,
                    )
                )

                # Peak alpha power and COG (global, each channel, somato-sensory)
                freqRange = (psd.freqs >= freq_bands["alpha"][0]) & (
                    psd.freqs <= freq_bands["alpha"][1]
                )

                # Per-trial features (new behavior)
                # Get data shape: (n_trials, n_channels, n_freqs)
                psd_data = psd.get_data()
                n_trials = psd_data.shape[0]

                # Initialize trial features for this condition
                trial_features = []

                for trial_idx in range(n_trials):
                    # Create trial row
                    trial_row = {
                        "participant_id": sub_num,
                        "session": session,
                        "trial_id": trial_idx,
                        "condition": sanitized_name,
                        "task": task,
                        "file_name": p,
                    }

                    # Average spectrum across channels for this trial
                    avgpow_trial = psd_data[trial_idx].mean(axis=0)
                    peak, prop = scipy.signal.find_peaks(avgpow_trial[freqRange])

                    # If more than one peak, keep the highest
                    if len(peak) > 1:
                        peak = peak[np.argmax(avgpow_trial[freqRange][peak])]
                        used_max_global = 0
                    elif len(peak) == 0:
                        # Use this workaround if no peak is found, use the max
                        peak = np.argmax(avgpow_trial[freqRange])
                        used_max_global = 1
                    else:
                        used_max_global = 0

                    # Store alpha peak features
                    trial_row["alpha_peak_global"] = psd.freqs[freqRange][peak]
                    trial_row["alpha_peak_global_usedmax"] = used_max_global

                    # Center of gravity
                    trial_row["alpha_cog_global"] = np.sum(
                        np.multiply(avgpow_trial[freqRange], psd.freqs[freqRange])
                    ) / np.sum(avgpow_trial[freqRange])

                    # Same for somatosensory channels
                    if somato_chans:
                        somato_data = psd.copy().pick(somato_chans).get_data()
                        avgpow_somato_trial = somato_data[trial_idx].mean(axis=0)
                        peak, prop = scipy.signal.find_peaks(
                            avgpow_somato_trial[freqRange]
                        )

                        if len(peak) > 1:
                            peak = peak[np.argmax(avgpow_somato_trial[freqRange][peak])]
                            used_max_somato = 0
                        elif len(peak) == 0:
                            peak = np.argmax(avgpow_somato_trial[freqRange])
                            used_max_somato = 1
                        else:
                            used_max_somato = 0

                        trial_row["alpha_peak_somato"] = psd.freqs[freqRange][peak]
                        trial_row["alpha_peak_somato_usedmax"] = used_max_somato
                        trial_row["alpha_cog_somato"] = np.sum(
                            np.multiply(
                                avgpow_somato_trial[freqRange], psd.freqs[freqRange]
                            )
                        ) / np.sum(avgpow_somato_trial[freqRange])

                    # Compute ROI features for alpha band (per-trial)
                    # Use configurable ROI channels or default if not provided
                    if roi_channels is None:
                        roi_channels = {
                            # Somatosensory Cortex (S1/S2) - pain localization and intensity coding
                            "somato": [
                                "C3",
                                "C4",
                                "CP3",
                                "CP4",
                                "C1",
                                "C2",
                                "C5",
                                "C6",
                                "CP1",
                                "CP2",
                                "CP5",
                                "CP6",
                                "Cz",
                                "CPz",
                            ],
                            # Prefrontal Cortex - attention, working memory, top-down modulation
                            "prefrontal": [
                                "Fp1",
                                "Fp2",
                                "AF3",
                                "AF4",
                                "AF7",
                                "AF8",
                                "AFz",
                            ],
                            # Frontal Cortex - cognitive control, motor planning
                            "frontal": [
                                "Fz",
                                "F1",
                                "F2",
                                "F3",
                                "F4",
                                "F5",
                                "F6",
                                "F7",
                                "F8",
                                "FC1",
                                "FC2",
                                "FC3",
                                "FC4",
                                "FC5",
                                "FC6",
                            ],
                            # Parietal Cortex - body representation, sensory integration
                            "parietal": [
                                "Pz",
                                "P1",
                                "P2",
                                "P3",
                                "P4",
                                "P5",
                                "P6",
                                "P7",
                                "P8",
                                "CP1",
                                "CP2",
                                "CP3",
                                "CP4",
                                "CPz",
                            ],
                            # Temporal Cortex - auditory processing, associative memory
                            "temporal": [
                                "T7",
                                "T8",
                                "TP7",
                                "TP8",
                                "TP9",
                                "TP10",
                                "FT7",
                                "FT8",
                                "FT9",
                                "FT10",
                            ],
                            # Occipital Cortex - visual perception
                            "occipital": [
                                "O1",
                                "O2",
                                "Oz",
                                "PO3",
                                "PO4",
                                "PO7",
                                "PO8",
                                "POz",
                            ],
                            # Midline - central integration (all cortical, no deep proxies)
                            "midline": [
                                "Fpz",
                                "AFz",
                                "Fz",
                                "FCz",
                                "Cz",
                                "CPz",
                                "Pz",
                                "POz",
                                "Oz",
                            ],
                        }

                    for roi_name, roi_chans in roi_channels.items():
                        # Check if any ROI channels are available in the data
                        available_roi_chans = [
                            ch for ch in roi_chans if ch in psd.ch_names
                        ]
                        if available_roi_chans:
                            roi_data = psd.copy().pick(available_roi_chans).get_data()
                            avgpow_roi_trial = roi_data[trial_idx].mean(axis=0)
                            peak, prop = scipy.signal.find_peaks(
                                avgpow_roi_trial[freqRange]
                            )
                            if len(peak) > 1:
                                peak = peak[
                                    np.argmax(avgpow_roi_trial[freqRange][peak])
                                ]
                                used_max_roi = 0
                            elif len(peak) == 0:
                                peak = np.argmax(avgpow_roi_trial[freqRange])
                                used_max_roi = 1
                            else:
                                used_max_roi = 0

                            trial_row[f"alpha_peak_{roi_name}"] = psd.freqs[freqRange][
                                peak
                            ]
                            trial_row[f"alpha_peak_{roi_name}_usedmax"] = used_max_roi
                            trial_row[f"alpha_cog_{roi_name}"] = np.sum(
                                np.multiply(
                                    avgpow_roi_trial[freqRange], psd.freqs[freqRange]
                                )
                            ) / np.sum(avgpow_roi_trial[freqRange])

                    # --- NEW: Compute ROI power for all frequency bands (per-trial) ---
                    for band in freq_bands:
                        curr_freqRange = (psd.freqs >= freq_bands[band][0]) & (
                            psd.freqs <= freq_bands[band][1]
                        )
                        for roi_name, roi_chans in roi_channels.items():
                            available_roi_chans = [
                                ch for ch in roi_chans if ch in psd.ch_names
                            ]
                            if available_roi_chans:
                                roi_data = (
                                    psd.copy().pick(available_roi_chans).get_data()
                                )
                                # Average across channels for this trial, then sum power in band
                                band_power = np.sum(
                                    roi_data[trial_idx][:, curr_freqRange].mean(axis=0)
                                )
                                trial_row[f"{band}_power_{roi_name}"] = band_power

                    # FOOOF processing for this trial
                    #############################################################
                    # Initialize model object
                    fm = SpectralModel(verbose=False)

                    # Define frequency range across which to model the spectrum
                    freq_range = [psd_freqmin, psd_freqmax]

                    # Parameterize the power spectrum for this trial
                    # Average spectrum across channels for this trial
                    power_spectrum = psd_data[trial_idx].mean(axis=0)
                    fm.fit(psd.freqs, power_spectrum, freq_range)

                    # Store FOOOF parameters for this trial
                    trial_row["foof_offset"] = fm.aperiodic_params_[0]
                    trial_row["foof_exponent"] = fm.aperiodic_params_[1]

                    # Save FOOOF model for this trial (only for first trial to avoid too many files)
                    if trial_idx == 0:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fm.save(
                                os.path.join(
                                    subject_eeg_dir,
                                    f"{sub_num}_task-{task}_{session_file}foof_{sanitized_name}.h5",
                                )
                            )
                            fm.save_report(
                                os.path.join(
                                    subject_eeg_dir,
                                    f"{sub_num}_task-{task}_{session_file}foof_report_{sanitized_name}.jpg",
                                )
                            )

                    trial_features.append(trial_row)

                # Add trial features to the main dataframe
                trial_df = pd.DataFrame(trial_features)
                features_frame = pd.concat(
                    [features_frame, trial_df], ignore_index=True
                )

                # Source space analysis and connectivity for this condition (per-trial)
                if compute_sourcespace_features and compute_connectivity:
                    #############################################################
                    # Source space and connectivity computation
                    #############################################################
                    msg = f"Computing features - source space and connectivity for condition: {sanitized_name}"
                    logger.info(
                        **gen_log_kwargs(
                            message=msg,
                            subject=sub_num.replace("sub-", ""),
                            session=session,
                        )
                    )

                    # For connectivity, use the condition-specific epochs
                    clean_epo_connectivity = (
                        epochs_for_condition.copy().set_eeg_reference(projection=True)
                    )

                    # NOTE maybe should download instead
                    pos = pd.read_csv(sourcecoords_file)

                    pos_coord = dict()

                    # Divide to convert mm to m
                    pos_coord["rr"] = np.array([pos["R"], pos["A"], pos["S"]]).T / 1000
                    pos_coord["nn"] = np.array([pos["R"], pos["A"], pos["S"]]).T / 1000
                    labels = pos["ROI Name"]

                    # Setup the source space
                    src = mne.setup_volume_source_space(
                        "fsaverage", pos=pos_coord, verbose=False
                    )

                    # Get standard bem model
                    bem = os.path.join(
                        fetch_fsaverage(), "bem", "fsaverage-5120-5120-5120-bem-sol.fif"
                    )

                    # Make forward model
                    forward = mne.make_forward_solution(
                        clean_epo_connectivity.info,
                        src=src,
                        trans="fsaverage",
                        bem=bem,
                        eeg=True,
                    )

                    # Bandpass the data in the relevant frequency band
                    for band in freq_bands:
                        clean_epo_band = clean_epo_connectivity.copy().filter(
                            freq_bands[band][0], freq_bands[band][1], n_jobs=-1
                        )

                        # Compute the covariance matrix from the data
                        cov = mne.compute_covariance(
                            clean_epo_band,
                            method="empirical",
                            keep_sample_mean=False,
                            verbose=False,
                        )

                        filters = make_lcmv(
                            clean_epo_band.info,
                            forward,
                            cov,
                            reg=0.05,
                            pick_ori="max-power",
                            rank=None,
                        )

                        # Apply the spatial filter
                        stcs = apply_lcmv_epochs(clean_epo_band, filters)

                        # Compute connectivity PER TRIAL using robust MNE methods
                        # Initialize arrays to collect all trial connectivity matrices
                        all_aec_matrices = []
                        all_wpli_matrices = []

                        for trial_idx in range(len(stcs)):
                            # Get single trial source estimate
                            stc_trial = stcs[trial_idx]

                            # For AEC, apply hilbert transform to get envelope
                            stc_hilbert = stc_trial.copy().apply_hilbert(envelope=True)
                            vtcs_trial = stc_hilbert.data[np.newaxis, :, :]

                            # Compute AEC for this single trial using proper MNE method
                            con_aec_trial = envelope_correlation(
                                data=vtcs_trial, orthogonalize="pairwise", verbose=False
                            )
                            con_aec_trial = (
                                con_aec_trial.combine()
                            )  # Combine connectivity data over epochs

                            # Get the AEC matrix for this trial
                            con_aec_matrix_trial = con_aec_trial.get_data(
                                output="dense"
                            )[:, :, 0]
                            con_aec_matrix_trial /= 0.577  # normalization because of under-estimation through orthogonalization
                            con_aec_matrix_trial[np.triu_indices(len(labels), 0)] = (
                                np.nan
                            )

                            # Collect AEC matrix for this trial
                            if "aec" in connectivity_methods:
                                all_aec_matrices.append(con_aec_matrix_trial)

                            # For wPLI, we need to use proper MNE spectral_connectivity_epochs
                            # Since we can't compute wPLI on a single trial, we'll use a sliding window approach
                            # or compute connectivity with the single trial and a reference
                            # For now, we'll compute a simplified connectivity measure for wPLI
                            # We'll use the phase information from the single trial
                            if "wpli" in connectivity_methods:
                                stc_phase = stc_trial.copy().apply_hilbert(
                                    envelope=False
                                )
                                phase_data = np.angle(stc_phase.data)

                                # Compute phase difference matrix for this trial
                                n_vertices = phase_data.shape[0]
                                phase_diff_matrix = np.zeros((n_vertices, n_vertices))

                                for i in range(n_vertices):
                                    for j in range(i + 1, n_vertices):
                                        phase_diff = phase_data[i, :] - phase_data[j, :]
                                        # Compute phase locking value
                                        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                                        phase_diff_matrix[i, j] = plv
                                        phase_diff_matrix[j, i] = plv

                                # Mask upper triangle for consistency
                                phase_diff_matrix[np.triu_indices(len(labels), 0)] = (
                                    np.nan
                                )

                                # Collect wPLI matrix for this trial
                                all_wpli_matrices.append(phase_diff_matrix)

                            # Compute graph metrics for each connectivity matrix (per trial) using BCT
                            for conn_measure in connectivity_methods:
                                if conn_measure == "wpli":
                                    conn_matrix = phase_diff_matrix.copy()
                                else:
                                    conn_matrix = con_aec_matrix_trial.copy()

                                # Threshold by the configurable percentage of the connectivity values
                                sortedValues = np.sort(np.abs(conn_matrix.flatten()))
                                # Remove NaNs
                                sortedValues = sortedValues[~np.isnan(sortedValues)]
                                # Get the threshold
                                threshold = sortedValues[
                                    int(connectivity_threshold * len(sortedValues))
                                ]
                                # Binarize the matrix
                                adjacency_matrix = np.abs(conn_matrix) >= threshold

                                # Compute graph metrics using BCT (robust, validated methods)
                                # Degree - Number of connections of each node
                                degree = bct.degree.degrees_und(adjacency_matrix)
                                # Clustering coefficient - The percentage of existing triangles surrounding
                                cc = bct.clustering.clustering_coef_bu(adjacency_matrix)
                                # Global clustering coefficient
                                gcc = np.mean(cc)
                                # Characteristic path length
                                distance = bct.distance.distance_bin(adjacency_matrix)
                                cpl = bct.distance.charpath(distance, 0, 0)[0]
                                # Global efficiency - The average of the inverse shortest path between two points
                                geff = bct.efficiency.efficiency_bin(adjacency_matrix)

                                # Store graph metrics in features frame for this specific trial
                                trial_row_idx = (
                                    len(features_frame)
                                    - len(trial_features)
                                    + trial_idx
                                )
                                if "density" in graph_metrics:
                                    features_frame.loc[
                                        trial_row_idx, f"{band}_{conn_measure}_density"
                                    ] = np.mean(degree)
                                if "clustering" in graph_metrics:
                                    features_frame.loc[
                                        trial_row_idx,
                                        f"{band}_{conn_measure}_clustering",
                                    ] = gcc
                                if "path_length" in graph_metrics:
                                    features_frame.loc[
                                        trial_row_idx,
                                        f"{band}_{conn_measure}_path_length",
                                    ] = cpl
                                if "efficiency" in graph_metrics:
                                    features_frame.loc[
                                        trial_row_idx,
                                        f"{band}_{conn_measure}_efficiency",
                                    ] = geff

                                # Add small-worldness for per-trial (same as old pipeline)
                                # Compute small-worldness for this trial
                                randN = bct.makerandCIJ_und(
                                    len(adjacency_matrix),
                                    int(np.floor(np.sum(adjacency_matrix) / 2)),
                                )
                                gcc_rand = np.mean(
                                    bct.clustering.clustering_coef_bu(randN)
                                )
                                cpl_rand = bct.distance.charpath(
                                    bct.distance.distance_bin(randN), 0, 0
                                )[0]
                                smallworldness = (gcc / gcc_rand) / (cpl / cpl_rand)
                                features_frame.loc[
                                    trial_row_idx,
                                    f"{band}_{conn_measure}_smallworldness",
                                ] = smallworldness

                        # Save ALL trial connectivity matrices as 3D arrays (trials x vertices x vertices)
                        if "aec" in connectivity_methods and all_aec_matrices:
                            all_aec_matrices = np.stack(all_aec_matrices, axis=0)
                            np.save(
                                os.path.join(
                                    subject_eeg_dir,
                                    f"{sub_num}_task-{task}_{session_file}connectivity_aec_{band}{condition_suffix}_all_trials.npy",
                                ),
                                all_aec_matrices,
                            )

                        if "wpli" in connectivity_methods and all_wpli_matrices:
                            all_wpli_matrices = np.stack(all_wpli_matrices, axis=0)
                            np.save(
                                os.path.join(
                                    subject_eeg_dir,
                                    f"{sub_num}_task-{task}_{session_file}connectivity_wpli_{band}{condition_suffix}_all_trials.npy",
                                ),
                                all_wpli_matrices,
                            )

                        # Save the labels (only once per condition)
                        np.save(
                            os.path.join(
                                subject_eeg_dir,
                                f"{sub_num}_task-{task}_{session_file}connectivity_labels{condition_suffix}.npy",
                            ),
                            labels,
                        )

                # PSD plots for this condition (only once per condition)
                #############################################################
                fig = psd.plot(show=False)
                fig.savefig(
                    os.path.join(
                        subject_eeg_dir,
                        f"{sub_num}_task-{task}_{session_file}psd{condition_suffix}.jpg",
                    )
                )

                # Average across both trials and channels to get 1D array for plotting
                dat = psd.get_data().mean(
                    axis=(0, 1)
                )  # Average across trials and channels
                plt.figure()
                plt.plot(psd.freqs, dat, label=f"Average PSD - {sanitized_name}")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Power")
                plt.title(f"Average PSD - {sanitized_name}")
                plt.legend()
                plt.savefig(
                    os.path.join(
                        subject_eeg_dir,
                        f"{sub_num}_task-{task}_{session_file}psd_bands{condition_suffix}.jpg",
                    )
                )
                plt.close("all")

        # For rest task, process the PSD data
        if task_is_rest:
            # Use configurable ROI channels or default if not provided
            if roi_channels is None:
                roi_channels = {
                    # Somatosensory Cortex (S1/S2) - pain localization and intensity coding
                    "somato": [
                        "C3",
                        "C4",
                        "CP3",
                        "CP4",
                        "C1",
                        "C2",
                        "C5",
                        "C6",
                        "CP1",
                        "CP2",
                        "CP5",
                        "CP6",
                        "Cz",
                        "CPz",
                    ],
                    # Prefrontal Cortex - attention, working memory, top-down modulation
                    "prefrontal": ["Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "AFz"],
                    # Frontal Cortex - cognitive control, motor planning
                    "frontal": [
                        "Fz",
                        "F1",
                        "F2",
                        "F3",
                        "F4",
                        "F5",
                        "F6",
                        "F7",
                        "F8",
                        "FC1",
                        "FC2",
                        "FC3",
                        "FC4",
                        "FC5",
                        "FC6",
                    ],
                    # Parietal Cortex - body representation, sensory integration
                    "parietal": [
                        "Pz",
                        "P1",
                        "P2",
                        "P3",
                        "P4",
                        "P5",
                        "P6",
                        "P7",
                        "P8",
                        "CP1",
                        "CP2",
                        "CP3",
                        "CP4",
                        "CPz",
                    ],
                    # Temporal Cortex - auditory processing, associative memory
                    "temporal": [
                        "T7",
                        "T8",
                        "TP7",
                        "TP8",
                        "TP9",
                        "TP10",
                        "FT7",
                        "FT8",
                        "FT9",
                        "FT10",
                    ],
                    # Occipital Cortex - visual perception
                    "occipital": ["O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8", "POz"],
                    # Midline - central integration (all cortical, no deep proxies)
                    "midline": [
                        "Fpz",
                        "AFz",
                        "Fz",
                        "FCz",
                        "Cz",
                        "CPz",
                        "Pz",
                        "POz",
                        "Oz",
                    ],
                }

            # Peak alpha power and COG (global, each channel, somato-sensory, and ROIs)
            #############################################################
            msg = "Computing features - Peak alpha for rest task."
            logger.info(
                **gen_log_kwargs(
                    message=msg, subject=sub_num.replace("sub-", ""), session=session
                )
            )

            # Peak alpha power and COG (global, each channel, somato-sensory)
            freqRange = (psd.freqs >= freq_bands["alpha"][0]) & (
                psd.freqs <= freq_bands["alpha"][1]
            )

            # Per-subject features (original behavior)
            # Average spectrum (across channels)
            avgpow = psd.get_data().mean(axis=0)
            peak, prop = scipy.signal.find_peaks(avgpow[freqRange])
            # If more than one peak, keep the highest
            if len(peak) > 1:
                peak = peak[np.argmax(avgpow[freqRange][peak])]
                used_max_global = 0
            elif len(peak) == 0:
                # Use this workaround if no peak is found, use the max
                # and flag it as a potential problem
                peak = np.argmax(avgpow[freqRange])
                used_max_global = 1
            else:
                used_max_global = 0

            # Store features with condition-specific keys
            alpha_key_prefix = "alpha_peak_global"
            features_frame.loc[(sub_num, session), alpha_key_prefix] = psd.freqs[
                freqRange
            ][peak]
            features_frame.loc[(sub_num, session), f"{alpha_key_prefix}_usedmax"] = (
                used_max_global
            )

            # Center of gravity
            cog_key_prefix = "alpha_cog_global"
            features_frame.loc[(sub_num, session), cog_key_prefix] = np.sum(
                np.multiply(avgpow[freqRange], psd.freqs[freqRange])
            ) / np.sum(avgpow[freqRange])

            # Same for somatosensory channels
            if somato_chans:
                avgpowe_samato = psd.copy().pick(somato_chans).get_data().mean(axis=0)
                peak, prop = scipy.signal.find_peaks(avgpowe_samato[freqRange])
                if len(peak) > 1:
                    peak = peak[np.argmax(avgpowe_samato[freqRange][peak])]
                    used_max_somato = 0
                elif len(peak) == 0:
                    # Use this workaround if no peak is found, use the max
                    peak = np.argmax(avgpow[freqRange])
                    used_max_somato = 1
                else:
                    used_max_somato = 0

                alpha_somato_key = "alpha_peak_somato"
                features_frame.loc[(sub_num, session), alpha_somato_key] = psd.freqs[
                    freqRange
                ][peak]
                features_frame.loc[
                    (sub_num, session), f"{alpha_somato_key}_usedmax"
                ] = used_max_somato

                # Center of gravity
                cog_somato_key = "alpha_cog_somato"
                features_frame.loc[(sub_num, session), cog_somato_key] = np.sum(
                    np.multiply(avgpowe_samato[freqRange], psd.freqs[freqRange])
                ) / np.sum(avgpowe_samato[freqRange])

            # Compute ROI features for alpha band
            for roi_name, roi_chans in roi_channels.items():
                # Check if any ROI channels are available in the data
                available_roi_chans = [ch for ch in roi_chans if ch in psd.ch_names]
                if available_roi_chans:
                    avgpow_roi = (
                        psd.copy().pick(available_roi_chans).get_data().mean(axis=0)
                    )
                    peak, prop = scipy.signal.find_peaks(avgpow_roi[freqRange])
                    if len(peak) > 1:
                        peak = peak[np.argmax(avgpow_roi[freqRange][peak])]
                        used_max_roi = 0
                    elif len(peak) == 0:
                        peak = np.argmax(avgpow_roi[freqRange])
                        used_max_roi = 1
                    else:
                        used_max_roi = 0

                    alpha_roi_key = f"alpha_peak_{roi_name}"
                    features_frame.loc[(sub_num, session), alpha_roi_key] = psd.freqs[
                        freqRange
                    ][peak]
                    features_frame.loc[
                        (sub_num, session), f"{alpha_roi_key}_usedmax"
                    ] = used_max_roi

                    # Center of gravity
                    cog_roi_key = f"alpha_cog_{roi_name}"
                    features_frame.loc[(sub_num, session), cog_roi_key] = np.sum(
                        np.multiply(avgpow_roi[freqRange], psd.freqs[freqRange])
                    ) / np.sum(avgpow_roi[freqRange])

            # --- NEW: Compute ROI power for all frequency bands (rest) ---
            for band in freq_bands:
                curr_freqRange = (psd.freqs >= freq_bands[band][0]) & (
                    psd.freqs <= freq_bands[band][1]
                )
                for roi_name, roi_chans in roi_channels.items():
                    available_roi_chans = [ch for ch in roi_chans if ch in psd.ch_names]
                    if available_roi_chans:
                        avgpow_roi = (
                            psd.copy().pick(available_roi_chans).get_data().mean(axis=0)
                        )
                        band_power = np.sum(avgpow_roi[curr_freqRange])
                        features_frame.loc[
                            (sub_num, session), f"{band}_power_{roi_name}"
                        ] = band_power

        # For rest task, process the PSD data for all channels
        if task_is_rest:
            # Same for all channels while we are at it
            peak_alpha_all_chans = pd.DataFrame(
                columns=["peak", "cog"], index=clean_epo.ch_names
            )
            for ch in clean_epo.ch_names:
                avgpow_ch = psd.copy().pick(ch).get_data().mean(axis=0)
                peak, prop = scipy.signal.find_peaks(avgpow_ch[freqRange])
                if len(peak) > 1:
                    peak = peak[np.argmax(avgpow_ch[freqRange][peak])]
                    used_max = 0
                elif len(peak) == 0:
                    # Use this workaround if no peak is found, use the max
                    # and flag it as a potential problem
                    peak = np.argmax(avgpow_ch[freqRange])
                    used_max = 1
                else:
                    used_max = 0
                peak_alpha_all_chans.loc[ch, "peak"] = psd.freqs[freqRange][peak]
                peak_alpha_all_chans.loc[ch, "peak_usedmax"] = used_max

                peak_alpha_all_chans.loc[ch, "cog"] = np.sum(
                    np.multiply(avgpow_ch[freqRange], psd.freqs[freqRange])
                ) / np.sum(avgpow_ch[freqRange])

            peak_alpha_all_chans.to_csv(
                os.path.join(
                    subject_eeg_dir,
                    f"{sub_num}_task-{task}_{session_file}peak_alpha_all_chans.tsv",
                ),
                sep="\t",
                index=True,
            )

            #############################################################
            # Average power in canonical bands
            #############################################################
            msg = "Computing features - Power and 1/f for rest task."
            logger.info(
                **gen_log_kwargs(
                    message=msg, subject=sub_num.replace("sub-", ""), session=session
                )
            )

            # Per-subject features (original behavior)
            # Average power in canonical bands
            for band in freq_bands:
                curr_freqRange = (psd.freqs >= freq_bands[band][0]) & (
                    psd.freqs <= freq_bands[band][1]
                )
                band_key = f"{band}_power_global"
                features_frame.loc[(sub_num, session), band_key] = np.sum(
                    psd.get_data()[:, curr_freqRange].mean(axis=1)
                )

                # Same in somatosensory channels
                if somato_chans:
                    band_somato_key = f"{band}_power_somato"
                    features_frame.loc[(sub_num, session), band_somato_key] = np.sum(
                        psd.copy()
                        .pick(somato_chans)
                        .get_data()[:, curr_freqRange]
                        .mean(axis=1)
                    )

                # Compute ROI power features for all frequency bands
                for roi_name, roi_chans in roi_channels.items():
                    # Check if any ROI channels are available in the data
                    available_roi_chans = [ch for ch in roi_chans if ch in psd.ch_names]
                    if available_roi_chans:
                        band_roi_key = f"{band}_power_{roi_name}"
                        features_frame.loc[(sub_num, session), band_roi_key] = np.sum(
                            psd.copy()
                            .pick(available_roi_chans)
                            .get_data()[:, curr_freqRange]
                            .mean(axis=1)
                        )

                for trial_idx in range(n_trials):
                    # Global power for this trial
                    trial_power_global = np.sum(
                        psd_data[trial_idx, :, curr_freqRange].mean(axis=1)
                    )
                    features_frame.loc[
                        last_trial_idx + trial_idx, f"{band}_power_global"
                    ] = trial_power_global

                    # Somatosensory power for this trial
                    if somato_chans:
                        somato_data = psd.copy().pick(somato_chans).get_data()
                        trial_power_somato = np.sum(
                            somato_data[trial_idx, :, curr_freqRange].mean(axis=1)
                        )
                        features_frame.loc[
                            last_trial_idx + trial_idx, f"{band}_power_somato"
                        ] = trial_power_somato

                    # Compute ROI power features for all frequency bands (per-trial)
                    for roi_name, roi_chans in roi_channels.items():
                        # Check if any ROI channels are available in the data
                        available_roi_chans = [
                            ch for ch in roi_chans if ch in psd.ch_names
                        ]
                        if available_roi_chans:
                            roi_data = psd.copy().pick(available_roi_chans).get_data()
                            trial_power_roi = np.sum(
                                roi_data[trial_idx, :, curr_freqRange].mean(axis=1)
                            )
                            features_frame.loc[
                                last_trial_idx + trial_idx, f"{band}_power_{roi_name}"
                            ] = trial_power_roi

            # Source space analysis for this condition
            if compute_sourcespace_features:
                #############################################################
                # Source space
                #############################################################
                msg = "Computing features - source space for rest task"
                logger.info(
                    **gen_log_kwargs(
                        message=msg,
                        subject=sub_num.replace("sub-", ""),
                        session=session,
                    )
                )

                # For connectivity, use the condition-specific epochs
                clean_epo_connectivity = epochs_for_condition.copy().set_eeg_reference(
                    projection=True
                )

                # NOTE maybe should download instead
                pos = pd.read_csv(sourcecoords_file)

                pos_coord = dict()

                # Divide to convert mm to m
                pos_coord["rr"] = np.array([pos["R"], pos["A"], pos["S"]]).T / 1000
                pos_coord["nn"] = np.array([pos["R"], pos["A"], pos["S"]]).T / 1000
                labels = pos["ROI Name"]

                # Setup the source space
                src = mne.setup_volume_source_space(
                    "fsaverage", pos=pos_coord, verbose=False
                )

                # Get standard bem model
                bem = os.path.join(
                    fetch_fsaverage(), "bem", "fsaverage-5120-5120-5120-bem-sol.fif"
                )

                # Make forward model
                forward = mne.make_forward_solution(
                    clean_epo_connectivity.info,
                    src=src,
                    trans="fsaverage",
                    bem=bem,
                    eeg=True,
                )

                # Bandpass the data in the relevant frequency band
                graph_measures = dict()
                for band in freq_bands:
                    clean_epo_band = clean_epo_connectivity.copy().filter(
                        freq_bands[band][0], freq_bands[band][1], n_jobs=-1
                    )

                    # Compute the covariance matrix from the data
                    cov = mne.compute_covariance(
                        clean_epo_band,
                        method="empirical",
                        keep_sample_mean=False,
                        verbose=False,
                    )

                    filters = make_lcmv(
                        clean_epo_band.info,
                        forward,
                        cov,
                        reg=0.05,
                        pick_ori="max-power",
                        rank=None,
                    )

                    # Apply the spatial filter
                    stcs = apply_lcmv_epochs(clean_epo_band, filters)

                    # Compute the connectivity via methods of Weighted PLI (wPLI) (only lower triangle)
                    if "wpli" in connectivity_methods:
                        con_wpli = spectral_connectivity_epochs(
                            data=stcs,
                            method="wpli",
                            mode="multitaper",
                            sfreq=clean_epo_band.info["sfreq"],
                            fmin=freq_bands[band][0],
                            fmax=freq_bands[band][1],
                            faverage=True,
                            n_jobs=-1,
                            verbose=False,
                        )

                        # Reshape the connectivity results back into a square matrix
                        con_wpli_matrix = (
                            con_wpli.get_data()
                            .squeeze()
                            .reshape(len(labels), len(labels))
                        )
                        # Mask the upper triangle
                        con_wpli_matrix[np.triu_indices(len(labels), 0)] = np.nan

                        # Save the connectivity matrix
                        connectivity_suffix = (
                            condition_suffix if not task_is_rest else ""
                        )
                        np.save(
                            os.path.join(
                                subject_eeg_dir,
                                f"{sub_num}_task-{task}_{session_file}connectivity_wpli_{band}{connectivity_suffix}.npy",
                            ),
                            con_wpli_matrix,
                        )

                    # Save the labels (only once per condition, not per band)
                    if band == list(freq_bands.keys())[0]:  # First band only
                        np.save(
                            os.path.join(
                                subject_eeg_dir,
                                f"{sub_num}_task-{task}_{session_file}connectivity_labels{connectivity_suffix}.npy",
                            ),
                            labels,
                        )

                    # Apply hilbert and stack data in 3D array with epochs x vertices x time
                    vtcs = []
                    for s in stcs:
                        vtcs.append(s.copy().apply_hilbert(envelope=True).data)

                    vtcs = np.stack(vtcs, axis=0)

                    # Compute the connectivity via methods of Amplitude Envelope Correlation (AEC)
                    if "aec" in connectivity_methods:
                        con_aec = envelope_correlation(
                            data=vtcs, orthogonalize="pairwise", verbose=False
                        )
                        con_aec = (
                            con_aec.combine()
                        )  # Combine connectivity data over epochs

                        # Retrieve the dense AEC matrix
                        con_aec_matrix = con_aec.get_data(output="dense")[:, :, 0]

                        con_aec_matrix /= 0.577  # normalization
                        con_aec_matrix[np.triu_indices(len(labels), 0)] = np.nan

                        # Save to file
                        np.save(
                            os.path.join(
                                subject_eeg_dir,
                                f"{sub_num}_task-{task}_{session_file}connectivity_aec_{band}{connectivity_suffix}.npy",
                            ),
                            con_aec_matrix,
                        )

                    # Plot the connectivity matrices for configured methods
                    if "wpli" in connectivity_methods:
                        plt.figure(figsize=(20, 20))
                        plt.imshow(con_wpli_matrix, cmap="viridis")
                        # Add labels to the matrix
                        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
                        plt.yticks(range(len(labels)), labels, fontsize=6)
                        plt.colorbar()
                        plt.title(f"wPLI Connectivity - {band}")
                        plt.savefig(
                            os.path.join(
                                subject_eeg_dir,
                                f"{sub_num}_task-{task}_{session_file}connectivitymatrix_wpli_{band}{connectivity_suffix}.jpg",
                            )
                        )
                        plt.close("all")

                    if "aec" in connectivity_methods:
                        # Plot the aec connectivity matrix
                        plt.figure(figsize=(20, 20))
                        plt.imshow(con_aec_matrix, cmap="viridis")
                        # Add labels to the matrix
                        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
                        plt.yticks(range(len(labels)), labels, fontsize=6)
                        plt.colorbar()
                        plt.title(f"AEC Connectivity - {band}")
                        plt.savefig(
                            os.path.join(
                                subject_eeg_dir,
                                f"{sub_num}_task-{task}_{session_file}connectivitymatrix_aec_{band}{connectivity_suffix}.jpg",
                            )
                        )
                    # Compute graph metrics for each connectivity matrix
                    graph_measures[band] = dict()
                    for conn_measure in connectivity_methods:
                        graph_measures[band][conn_measure] = dict()
                        if conn_measure == "wpli":
                            conn_matrix = con_wpli_matrix.copy()
                        else:
                            conn_matrix = con_aec_matrix.copy()

                        # Threshold by the configurable percentage of the connectivity values
                        sortedValues = np.sort(np.abs(conn_matrix.flatten()))
                        # Remove NaNs
                        sortedValues = sortedValues[~np.isnan(sortedValues)]
                        # Get the threshold
                        threshold = sortedValues[
                            int(connectivity_threshold * len(sortedValues))
                        ]
                        # Binarize the matrix
                        adjacency_matrix = np.abs(conn_matrix) >= threshold

                        # Plot the connectome
                        conn_matrix_plot = conn_matrix.copy()
                        # Remove nans
                        conn_matrix_plot[np.isnan(conn_matrix_plot)] = 0
                        # Make it symmetric
                        conn_matrix_plot = (
                            conn_matrix_plot
                            + conn_matrix_plot.T
                            - np.diag(np.diag(conn_matrix_plot))
                        )
                        plot_connectome(
                            conn_matrix_plot,
                            node_coords=pos_coord["rr"] * 1000,
                            edge_threshold="99%",
                            node_size=10,
                            title=f"{band} {conn_measure} thresholded at 99%",
                            colorbar=True,
                            edge_cmap="viridis",
                            edge_vmin=0,
                            edge_vmax=1,
                        )
                        plt.savefig(
                            os.path.join(
                                subject_eeg_dir,
                                f"{sub_num}_task-{task}_{session_file}connectome_{conn_measure}_{band}{connectivity_suffix}.jpg",
                            )
                        )
                        plt.close("all")

                        # Compute graph metrics using BCT (robust, validated methods)
                        # Degree - Number of connections of each node
                        degree = bct.degree.degrees_und(adjacency_matrix)
                        # Clustering coefficient - The percentage of existing triangles surrounding
                        cc = bct.clustering.clustering_coef_bu(adjacency_matrix)
                        # Global clustering coefficient
                        gcc = np.mean(cc)
                        # Characteristic path length
                        distance = bct.distance.distance_bin(adjacency_matrix)
                        cpl = bct.distance.charpath(distance, 0, 0)[0]
                        # Global efficiency - The average of the inverse shortest path between two points
                        geff = bct.efficiency.efficiency_bin(adjacency_matrix)

                        # Store graph metrics (same as old pipeline)
                        graph_measures[band][conn_measure]["threshold"] = threshold
                        graph_measures[band][conn_measure]["degree"] = degree
                        graph_measures[band][conn_measure]["cc"] = cc
                        graph_measures[band][conn_measure]["gcc"] = gcc
                        graph_measures[band][conn_measure]["cpl"] = cpl
                        graph_measures[band][conn_measure]["geff"] = geff

                        # Compute small-worldness (same as old pipeline)
                        randN = bct.makerandCIJ_und(
                            len(adjacency_matrix),
                            int(np.floor(np.sum(adjacency_matrix) / 2)),
                        )
                        gcc_rand = np.mean(bct.clustering.clustering_coef_bu(randN))
                        cpl_rand = bct.distance.charpath(
                            bct.distance.distance_bin(randN), 0, 0
                        )[0]
                        graph_measures[band][conn_measure]["smallworldness"] = (
                            gcc / gcc_rand
                        ) / (cpl / cpl_rand)

                        # Plot degree at each node (same as old pipeline)
                        plot_markers(
                            degree,
                            pos_coord["rr"] * 1000,
                            title="Degree - " + band + " - " + conn_measure,
                        )
                        plt.savefig(
                            os.path.join(
                                subject_eeg_dir,
                                f"{sub_num}_task-{task}_{session_file}degree_{band}_{conn_measure}{connectivity_suffix}.jpg",
                            )
                        )
                        plt.close("all")

                        # Plot clustering coefficient at each node (same as old pipeline)
                        plot_markers(
                            cc,
                            pos_coord["rr"] * 1000,
                            title="Clustering coefficient - "
                            + band
                            + " - "
                            + conn_measure,
                        )
                        plt.savefig(
                            os.path.join(
                                subject_eeg_dir,
                                f"{sub_num}_task-{task}_{session_file}cc_{band}_{conn_measure}{connectivity_suffix}.jpg",
                            )
                        )
                        plt.close("all")

                        # Store metrics for features frame (using density, clustering, path_length, efficiency names)
                        graph_measures[band][conn_measure]["density"] = np.mean(degree)
                        graph_measures[band][conn_measure]["clustering"] = gcc
                        graph_measures[band][conn_measure]["path_length"] = cpl
                        graph_measures[band][conn_measure]["efficiency"] = geff

                        # Store graph metrics in features frame (same as old pipeline)
                        if "density" in graph_metrics:
                            features_frame.loc[
                                (sub_num, session), f"{band}_{conn_measure}_density"
                            ] = graph_measures[band][conn_measure]["density"]
                        if "clustering" in graph_metrics:
                            features_frame.loc[
                                (sub_num, session), f"{band}_{conn_measure}_clustering"
                            ] = graph_measures[band][conn_measure]["clustering"]
                        if "path_length" in graph_metrics:
                            features_frame.loc[
                                (sub_num, session), f"{band}_{conn_measure}_path_length"
                            ] = graph_measures[band][conn_measure]["path_length"]
                        if "efficiency" in graph_metrics:
                            features_frame.loc[
                                (sub_num, session), f"{band}_{conn_measure}_efficiency"
                            ] = graph_measures[band][conn_measure]["efficiency"]

                        # Add small-worldness (same as old pipeline)
                        features_frame.loc[
                            (sub_num, session), f"{band}_{conn_measure}_smallworldness"
                        ] = graph_measures[band][conn_measure]["smallworldness"]

                # Save the graph measures
                graph_measures_suffix = condition_suffix if not task_is_rest else ""
                np.save(
                    os.path.join(
                        subject_eeg_dir,
                        f"{sub_num}_task-{task}_graph_measures{graph_measures_suffix}.npy",
                    ),
                    graph_measures,
                )

        # End of condition loop - Save the features frame
        if task_is_rest:
            # Save with index
            features_frame.to_csv(
                os.path.join(
                    subject_eeg_dir,
                    f"{sub_num}_task-{task}_{session_file}features_frame.tsv",
                ),
                sep="\t",
                index=True,
            )
        else:
            # For per-trial data, save without index
            features_frame.to_csv(
                os.path.join(
                    subject_eeg_dir,
                    f"{sub_num}_task-{task}_{session_file}features_frame.tsv",
                ),
                sep="\t",
                index=False,
            )
        plt.close("all")

        return features_frame

    # Use joblib to parallelize the process
    # For task-based analysis (task_is_rest=False), force n_jobs=1 to avoid pickling issues
    # with complex condition-based processing
    if n_jobs != 1 and task_is_rest:
        frames = Parallel(n_jobs=n_jobs)(
            delayed(_compute_features)(
                p,
                task=task,
                out_path=out_path,
                freq_bands=freq_bands,
                freq_res=freq_res,
                somato_chans=somato_chans,
                psd_freqmax=psd_freqmax,
                psd_freqmin=psd_freqmin,
                sourcecoords_file=sourcecoords_file,
                interpolate_bads=interpolate_bads,
                compute_sourcespace_features=compute_sourcespace_features,
                task_is_rest=task_is_rest,
                roi_channels=roi_channels,
            )
            for p in clean_epo_files
        )
    else:
        frames = []
        for p in clean_epo_files:
            frames.append(
                _compute_features(
                    p,
                    task=task,
                    out_path=out_path,
                    freq_bands=freq_bands,
                    freq_res=freq_res,
                    somato_chans=somato_chans,
                    psd_freqmax=psd_freqmax,
                    psd_freqmin=psd_freqmin,
                    sourcecoords_file=sourcecoords_file,
                    interpolate_bads=interpolate_bads,
                    compute_sourcespace_features=compute_sourcespace_features,
                    task_is_rest=task_is_rest,
                    roi_channels=roi_channels,
                )
            )

    # Handle saving based on task_is_rest
    if task_is_rest:
        # Original behavior: concatenate all frames into a single dataframe
        if len(frames) == 1:
            all_frames = frames[0]
        else:
            # Concatenate all frames
            all_frames = pd.concat(frames)
        all_frames.to_csv(
            os.path.join(out_path, f"task-{task}_features_frame.tsv"),
            sep="\t",
            index=True,
        )
    else:
        # New behavior: save individual subject files
        for frame in frames:
            if frame is not None and not frame.empty:
                # Extract subject ID from the first row
                subject_id = frame.iloc[0]["participant_id"]
                # Save individual subject file
                frame.to_csv(
                    os.path.join(
                        out_path, f"{subject_id}_task-{task}_features_frame.tsv"
                    ),
                    sep="\t",
                    index=False,
                )
        # Also create a pooled file for convenience
        if len(frames) > 1:
            all_frames = pd.concat(frames, ignore_index=True)
            all_frames.to_csv(
                os.path.join(out_path, f"task-{task}_features_frame_pooled.tsv"),
                sep="\t",
                index=False,
            )


# plots summarizing the features
def group_plot_features(out_path, task):
    """
    Plot the features for each subject.
    """
    # Initialize mne report
    report = mne.Report()
    report.add_section("Peak alpha power and COG", level=1)

    # Load the features
    features = pd.read_csv(
        os.path.join(out_path, f"task-{task}_features_frame.tsv"), sep="\t"
    )

    # Distribution plot of alpha peak using a raincloud plot
    fig = plt.figure(figsize=(10, 5))
    sns.violinplot(
        features["alpha_cog_global"],
        label="Global",
    )
    sns.swarmplot(
        features["alpha_cog_global"],
        label="Global",
    )
    plt.figure(figsize=(10, 5))
    sns.histplot(
        features["alpha_peak_somato"],
        bins=20,
        kde=True,
        color="red",
        label="Somatosensory",
    )

    plt.figure(figsize=(10, 5))

    sns.histplot(
        features["alpha_peak_somato"],
        bins=20,
        kde=True,
        color="red",
        label="Somatosensory",
    )
    plt.title("Alpha peak frequency distribution at global and somatosensory channels")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Density")
    plt.legend()


def custom_tfr(
    pipeline_path,
    task,
    freqs=np.arange(1, 100, 1),
    n_cycles=None,
    subjects="all",
    decim=1,
    n_jobs=1,
    return_itc=True,
    interpolate_bads=True,
    average=False,
    return_average=False,
    crop=None,
):
    """
    Custom TFR function to compute time-frequency representations.
    Parameters
    ----------
    freqs : array-like
        Frequencies to compute the TFR for.
    n_cycles : array-like
        Number of cycles for each frequency.
    decim : int
        Decimation factor for the TFR.
    n_jobs : int
        Number of jobs to run in parallel. This is for the TFR computation. Participants are not parallelized due to the computing demand of the TFR.
    return_itc : bool
        Whether to return the inter-trial coherence (ITC) in addition to power.
    interpolate_bads : bool
        Whether to interpolate bad channels before computing the TFR.
    average : bool
        Whether to average the TFR across epochs.
    return_average : bool
        Whether to return the average TFR across epochs in addition to the single epoch TFRs (only if average is False).
    crop : tuple of float
        Time range to crop the TFR to. If None, no cropping is done.
    """
    # Get the clean epochs files
    clean_epo_files = list(
        set(
            str(f)
            for f in BIDSPath(
                root=pipeline_path,
                task=task,
                session=None,
                suffix="epo",
                processing="clean",
                extension=".fif",
                check=False,
            ).match()
        )
    )
    clean_epo_files.sort()

    # Keep only the subjects specified
    if subjects != "all":
        clean_epo_files = [
            f
            for f in clean_epo_files
            if get_entities_from_fname(f)["subject"] in subjects
        ]

    # If n_cycles is not specified, use the frequencies as n_cycles
    if n_cycles is None:
        n_cycles = freqs / 3.0
    logger.title(f"Custom step - Computing TFR in {len(clean_epo_files)} files.")

    def _compute_tfr(
        p,
        freqs=freqs,
        n_cycles=n_cycles,
        decim=decim,
        n_jobs=n_jobs,
        return_itc=return_itc,
        interpolate_bads=interpolate_bads,
        average=average,
        crop=crop,
        return_average=return_average,
        conditions=None,
    ):
        # Load the cleaned epochs
        epo = mne.read_epochs(p)

        if interpolate_bads:
            epo = epo.interpolate_bads()

        if conditions is not None:
            # Select only the conditions specified
            epo = epo[conditions]

        sub_num = get_entities_from_fname(p)["subject"]
        session = get_entities_from_fname(p)["session"]

        msg = "Computing TFR"
        # Get subject from file
        logger.info(**gen_log_kwargs(message=msg, subject=sub_num, session=session))

        # Compute the TFR
        if not average:
            # Compute the TFR for each epoch using the newer method
            power = epo.compute_tfr(
                method="morlet",
                freqs=freqs,
                n_cycles=n_cycles,
                decim=decim,
                n_jobs=n_jobs,
                return_itc=return_itc,  # Return ITC if specified
                average=False,  # Do NOT average across epochs
            )
            # If return_itc is True, power will be a tuple of (power, itc)
            if return_itc:
                power, itc = power

                if crop:
                    itc.crop(tmin=crop[0], tmax=crop[1], include_tmax=True)
                # Save ITC data as numpy array to preserve trial structure
                itc_data = itc.data
                np.save(p.replace("_proc-clean_epo.fif", "_itc_epo-tfr.npy"), itc_data)
                # Also save the ITC object for compatibility
                itc.save(
                    p.replace("_proc-clean_epo.fif", "_itc_epo-tfr.h5"), overwrite=True
                )

            # Save the power data as numpy array to preserve trial structure
            if crop:
                power.crop(tmin=crop[0], tmax=crop[1], include_tmax=True)

            power_data = power.data
            np.save(p.replace("_proc-clean_epo.fif", "_power_epo-tfr.npy"), power_data)
            # Also save the power object for compatibility
            power.save(
                p.replace("_proc-clean_epo.fif", "_power_epo-tfr.h5"),
                overwrite=True,
            )

            if False:  # Formerly if return_average:
                for cond in epo.event_id:
                    # Sanitize condition name
                    cond_save = cond.replace(" ", "").replace("-", "").replace("/", "")

                    # Select epochs for the condition
                    power_cond = power[cond]
                    # Average the TFR across epochs
                    power_cond = power_cond.average()
                    # Save the average TFR object
                    power_cond.save(
                        p.replace(
                            "_proc-clean_epo.fif", "_power_" + cond_save + "_avg-tfr.h5"
                        ),
                        overwrite=True,
                    )
                    if return_itc:
                        itc_cond = itc[cond]
                        # Average the ITC across epochs and re-assign
                        itc_cond = itc_cond.average()
                        # Save the average ITC object
                        itc_cond.save(
                            p.replace(
                                "_proc-clean_epo.fif",
                                "_itc_" + cond_save + "_avg-tfr.h5",
                            ),
                            overwrite=True,
                        )
        else:
            # This block now handles all averaged TFR generation
            for cond in epo.event_id:
                # Sanitize condition name
                cond_save = cond.replace(" ", "").replace("-", "").replace("/", "")
                # Select epochs for the condition
                epochs_cond = epo[cond]

                # Compute the TFR for the condition using the newer method
                power_cond = epochs_cond.compute_tfr(
                    method="morlet",
                    freqs=freqs,
                    n_cycles=n_cycles,
                    decim=decim,
                    n_jobs=n_jobs,
                    return_itc=return_itc,  # Return ITC if specified
                    average=True,  # Average across epochs
                )
                if return_itc:
                    power_cond, itc_cond = power_cond
                    if crop:
                        itc_cond.crop(tmin=crop[0], tmax=crop[1], include_tmax=True)
                    # Save ITC
                    itc_cond.save(
                        p.replace(
                            "_proc-clean_epo.fif", f"_itc_{cond_save}_avg-tfr.h5"
                        ),
                        overwrite=True,
                    )

                # If crop is specified, crop the power object
                if crop:
                    power_cond.crop(tmin=crop[0], tmax=crop[1], include_tmax=True)
                # Save the TFR object
                power_cond.save(
                    p.replace("_proc-clean_epo.fif", f"_power_{cond_save}_avg-tfr.h5"),
                    overwrite=True,
                )

        msg = "Done computing TFR"
        # Get subject from file
        logger.info(
            **gen_log_kwargs(message=msg, subject=sub_num, session=session, emoji="✅")
        )

    for p in clean_epo_files:
        _compute_tfr(
            p,
            freqs=freqs,
            n_cycles=n_cycles,
            decim=decim,
            n_jobs=n_jobs,
            return_itc=return_itc,
            interpolate_bads=interpolate_bads,
            average=average,
            crop=crop,
            return_average=return_average,
            conditions=None,
        )


def synchronize_bad_channels_across_runs(config_file):
    """
    Synchronize bad channels across all runs for each subject.
    
    This function reads all channel TSV files for each subject, identifies 
    the union of bad channels across all runs, and updates all channel files
    to have the same bad channels. This ensures compatibility with MNE functions
    that require matching bad channels across runs (e.g., concatenate_epochs).
    """
    from mne_bids_pipeline._logging import gen_log_kwargs, logger
    import pandas as pd
    import glob
    import os
    
    # Get config values
    bids_root = get_config_keyval(config_file, "bids_root")
    subjects_config = get_config_keyval(config_file, "subjects")
    task = get_config_keyval(config_file, "task")
    
    logger.info("🔄 Synchronizing bad channels across runs for each subject...")
    
    # Handle subjects = "all" case by discovering actual subject IDs
    if subjects_config == "all":
        # Find all subject directories in BIDS root
        subject_dirs = glob.glob(os.path.join(bids_root, "sub-*"))
        subjects = [os.path.basename(d).replace("sub-", "") for d in subject_dirs if os.path.isdir(d)]
        logger.info(f"📂 Discovered {len(subjects)} subjects: {subjects}")
    else:
        subjects = subjects_config
    
    for subject in subjects:
        # Find all channel TSV files for this subject and task
        pattern = os.path.join(bids_root, f"sub-{subject}", "eeg", f"sub-{subject}_task-{task}_*_channels.tsv")
        channel_files = glob.glob(pattern)
        
        if not channel_files:
            logger.warning(f"No channel files found for subject {subject}")
            continue
            
        logger.info(f"📋 Processing {len(channel_files)} channel files for sub-{subject}")
        
        # Collect all bad channels across runs
        all_bad_channels = set()
        channel_data = {}
        
        for file_path in channel_files:
            df = pd.read_csv(file_path, sep='\t')
            # Find bad channels (status == 'bad')
            bad_channels = df[df['status'] == 'bad']['name'].tolist()
            all_bad_channels.update(bad_channels)
            channel_data[file_path] = df
            
            run_info = os.path.basename(file_path).split('_')
            run_id = next((part for part in run_info if part.startswith('run-')), 'unknown')
            logger.info(f"  📁 {run_id}: Found {len(bad_channels)} bad channels: {bad_channels}")
        
        # Union of all bad channels
        unified_bad_channels = sorted(list(all_bad_channels))
        logger.info(f"🔗 Unified bad channels for sub-{subject}: {unified_bad_channels} (total: {len(unified_bad_channels)})")
        
        # Update all channel files with unified bad channels
        for file_path, df in channel_data.items():
            # Reset all channels to 'good' first
            df['status'] = 'good'
            # Mark unified bad channels as 'bad'
            df.loc[df['name'].isin(unified_bad_channels), 'status'] = 'bad'
            # Save updated file
            df.to_csv(file_path, sep='\t', index=False)
            
            run_info = os.path.basename(file_path).split('_')
            run_id = next((part for part in run_info if part.startswith('run-')), 'unknown')
            logger.info(f"  ✅ Updated {run_id} with {len(unified_bad_channels)} bad channels")
    
    logger.info("✅ Bad channel synchronization completed")


def run_pipeline_task(task, config_file):
    #########################################################
    # Update config file with task
    #########################################################
    update_config(config_file, {"task": task})

    #########################################################
    # Print some info
    #########################################################
    # Log the date
    from datetime import datetime

    logger.info(
        msg=f"👍 Running preprocessing pipeline for task: {task} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    #########################################################
    # Bad channels using pyprep (not yet implemented in mne-bids pipeline for EEG)
    #########################################################
    if get_config_keyval(config_file, "use_pyprep"):
        run_bads_detection(**get_specific_config(config_file, "pyprep"))
        # Synchronize bad channels across all runs for each subject
        synchronize_bad_channels_across_runs(config_file)

    #########################################################
    # First pass mne-bids pipeline to get ICA components and create the components.tsv file
    #########################################################
    if get_config_keyval(config_file, "use_icalabel"):
        # Use subprocess instead of sys.argv modification to avoid concurrency issues
        cmd = [
            sys.executable,
            "-c",
            "from mne_bids_pipeline._main import main; import sys; sys.argv = ['mne_bids_pipeline', '--config="
            + config_file
            + "', '--steps=init,preprocessing/_01_data_quality,preprocessing/_04_frequency_filter,preprocessing/_05_regress_artifact,preprocessing/_06a1_fit_ica']; main()",
        ]
        logger.info(f"Running MNE-BIDS pipeline command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Log both stdout and stderr for debugging
        if result.stdout:
            logger.info(f"MNE-BIDS pipeline stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"MNE-BIDS pipeline stderr: {result.stderr}")
            
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout if result.stdout else "No error output captured"
            logger.error(f"MNE-BIDS pipeline failed with return code {result.returncode}. Error output: {error_msg}")
            raise RuntimeError(f"MNE-BIDS pipeline step failed with return code {result.returncode}: {error_msg}")
    else:
        # If not using icalabel, use mne_bids_pipeline to find bad icas
        cmd = [
            sys.executable,
            "-c",
            "from mne_bids_pipeline._main import main; import sys; sys.argv = ['mne_bids_pipeline', '--config="
            + config_file
            + "', '--steps=init,preprocessing/_01_data_quality,preprocessing/_04_frequency_filter,preprocessing/_05_regress_artifact,preprocessing/_06a1_fit_ica,preprocessing/_06a2_find_ica_artifacts.py']; main()",
        ]
        logger.info(f"Running MNE-BIDS pipeline command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Log both stdout and stderr for debugging
        if result.stdout:
            logger.info(f"MNE-BIDS pipeline stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"MNE-BIDS pipeline stderr: {result.stderr}")
            
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout if result.stdout else "No error output captured"
            logger.error(f"MNE-BIDS pipeline failed with return code {result.returncode}. Error output: {error_msg}")
            raise RuntimeError(f"MNE-BIDS pipeline step failed with return code {result.returncode}: {error_msg}")

    #########################################################
    # Flag bad ICA components using mne_icalabel (not yet implemented in mne-bids pipeline for EEG)
    #########################################################
    if get_config_keyval(config_file, "use_icalabel"):
        run_ica_label(**get_specific_config(config_file, "icalabel"))

    #########################################################
    # Run mne-bids pipeline ICA after components are flagged for artifact, epochs and ptp rejection
    #########################################################
    # Use subprocess instead of sys.argv modification to avoid concurrency issues
    cmd = [
        sys.executable,
        "-c",
        "from mne_bids_pipeline._main import main; import sys; sys.argv = ['mne_bids_pipeline', '--config="
        + config_file
        + "', '--steps=preprocessing/_07_make_epochs,preprocessing/_08a_apply_ica,preprocessing/_09_ptp_reject']; main()",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"MNE-BIDS pipeline failed with error: {result.stderr}")
        raise RuntimeError(f"MNE-BIDS pipeline step failed: {result.stderr}")

    #########################################################
    # Collect preprocessing metrics
    #########################################################
    collect_preprocessing_stats(
        bids_path=get_config_keyval(config_file, "bids_root"),
        pipeline_path=get_config_keyval(config_file, "deriv_root"),
        task=task,
    )

    #########################################################
    # Extract rest features from the data and make some plots subject-wise
    #########################################################
    if get_config_keyval(config_file, "compute_features"):
        compute_features(**get_specific_config(config_file, "features"))

    #########################################################
    # Compute TFRs
    #########################################################
    if get_config_keyval(config_file, "custom_tfr"):
        custom_tfr(**get_specific_config(config_file, "custom_tfr"))

    # Done !
    logger.info(f"✅ Pipeline completed successfully for task: {task}.")


import textwrap
from pathlib import Path

from mne_bids_pipeline._config_template import create_template_config


def generate_config(path="example_config.py", mne_bids_type="minimal"):
    # Custom part of the config
    with open(path, "w") as f:
        f.write(
            "########################################################\n# This is a custom config file for the labmp EEG pipeline.\n###########################################################"
        )
        f.write(
            "\n\n# It includes the default config from mne-bids pipeline and some custom settings.\n\n"
        )
        f.write("# Imports\n")
        f.write(
            "import os\nimport numpy as np\nfrom mne_bids import BIDSPath, get_entities_from_fname\n"
        )
        f.write("\n# Global settings\n")
        f.write("# bids_root: Path to the BIDS dataset\n")
        f.write("bids_root = '/path/to/bids/dataset'\n")
        f.write(
            "# external_root: Path to the external data folder (e.g. Schaefer atlas)\n"
        )
        f.write("external_root = '/path/to/external/data'\n")
        f.write(
            "# log_type: how to log the messages from the pipeline. Use 'console' to print everything to the console or 'file' to log all to a file in the preprocessed folder (recommended)\n"
        )
        f.write("log_type = 'file'\n")
        f.write(
            "# use_pyprep: Set to True to use pyprep for bad channels detection. If false, no automated bad channels dection\n"
        )
        f.write("use_pyprep = True\n")
        f.write(
            "# use_icalabel: Set to True to use mne-icalabel for ICA component classification. If false, the mne-bids-pipeline default classification based on eog and ecg is used\n"
        )
        f.write("use_icalabel = True\n")
        f.write(
            "# use_custom_tfr: Set to True to use the custom TFR function to compute time-frequency representations. If false, no TFRs are computed\n"
        )
        f.write("custom_tfr = True\n")
        f.write(
            "# compute_features: Set to True to compute features after preprocessing. If false, no features are computed\n"
        )
        f.write("compute_features = True\n")
        f.write("# tasks_to_process: List of tasks to process.\n")
        f.write("tasks_to_process = []\n")
        f.write(
            "#config_validation: mne-bids-pipeline config validation. Leave to False because we use custom options.\n"
        )
        f.write("config_validation = False\n")
        f.write(
            "# subjects: List of subjects to process or 'all' to process all subjects.\n"
        )
        f.write("subjects = 'all'\n")
        f.write(
            "# sessions: List of sessions to process or leave empty to process all sessions.\n"
        )
        f.write("sessions = []\n")
        f.write(
            "#task: Task to process. This will be updated iteratively by the pipeline for each task. No need to change.\n"
        )
        f.write("task = ''\n")
        f.write(
            "# deriv_root: Path to the derivatives folder where the preprocessed data will be saved.\n"
        )
        f.write(
            "deriv_root = os.path.join(bids_root, 'derivatives', 'task_' + task, 'preprocessed')\n"
        )
        f.write(
            "# select_subjects: If True, only the subjects with a file for the current task will be processed. If False, pipeline will crash if missing task\n"
        )
        f.write("select_subjects = True\n")
        f.write("if select_subjects:\n")
        f.write(
            "    task_subs = list(set(str(f) for f in BIDSPath(root=bids_root, task=task, session=None, datatype='eeg', suffix='eeg', extension='vhdr').match()))\n"
        )
        f.write(
            "    task_subs = [get_entities_from_fname(f).get('subject') for f in task_subs]\n"
        )
        f.write("    if subjects != 'all':\n")
        f.write("        # If subjects is not 'all', filter the task_subs list\n")
        f.write("        subjects = [sub for sub in task_subs if sub in subjects]\n")
        f.write("    else:\n")
        f.write("        # If subjects is 'all', use all available subjects\n")
        f.write("        subjects = task_subs\n")
        f.write("\n")
        f.write(
            "########################################################\n# Options for pyprep bad channels detection\n###########################################################\n"
        )
        f.write(
            "# pyprep_bids_path: Path to the BIDS dataset for pyprep, do not change unless you want a different path from the bids files\n"
        )
        f.write("pyprep_bids_path = bids_root\n")
        f.write(
            "# pyprep_pipeline_path: Path to the derivatives folder where the preprocessed data will be saved for pyprep, do not change\n"
        )
        f.write("pyprep_pipeline_path = deriv_root\n")
        f.write("# pyprep_task: Task to process for pyprep, do not change\n")
        f.write("pyprep_task = task\n")
        f.write(
            "# pyprep_ransac: Set to True to use RANSAC for bad channels detection, False to use only the other methods\n"
        )
        f.write("pyprep_ransac = False\n")
        f.write(
            "# pyprep_repeats: Number of repeats for the bad channel detection. This can improve detection by removing very bad channels and iterating again\n"
        )
        f.write("pyprep_repeats = 3\n")
        f.write(
            "# pyprep_average_reref: Set to True to average rereference the data before bad channels detection, False to use the original data\n"
        )
        f.write("pyprep_average_reref = False\n")
        f.write(
            "# pyprep_file_extension: File extension to use for the data files, default is .vhdr for BrainVision files\n"
        )
        f.write("pyprep_file_extension = '.vhdr'\n")
        f.write(
            "# pyprep_montage: Montage to use for the data, default is easycap-M1 for BrainVision files\n"
        )
        f.write("pyprep_montage = 'easycap-M1'\n")
        f.write(
            "# pyprep_l_pass: Low pass filter frequency for the data, default is 100.0 Hz\n"
        )
        f.write("pyprep_l_pass = 100.0\n")
        f.write(
            "# pyprep_notch: Notch filter frequency for the data, default is 60.0 Hz\n"
        )
        f.write("pyprep_notch = 60.0\n")
        f.write(
            "# pyprep_consider_previous_bads: Set to True to consider previous bad channels in the data (e.g. visually identified), False to ignore and clear them (e.g. when re-running the pipeline)\n"
        )
        f.write("pyprep_consider_previous_bads = False\n")
        f.write(
            "# pyprep_rename_anot_dict: Dictionary to rename the annotations to the format expected by MNE (e.g. BAD_)\n"
        )
        f.write("pyprep_rename_anot_dict = None\n")
        f.write(
            "# pyprep_overwrite_chans_tsv: Set to True to overwrite the channels.tsv file with the bad channels detected by pyprep, False to keep the original file and create a second file. mne-bids-pipeline will only use original file so not recommended to set to False\n"
        )
        f.write("pyprep_overwrite_chans_tsv = True\n")
        f.write("# pyprep_n_jobs: Number of jobs to use for pyprep, default is 1\n")
        f.write("pyprep_n_jobs = 1\n")
        f.write(
            "# pyprep_subjects: List of subjects to process for pyprep, default is same as the rest of the pipeline\n"
        )
        f.write("pyprep_subjects = subjects\n")
        f.write(
            "# pyprep_delete_breaks: Set to True to delete breaks in the data (only for this operation, the data file is not modified), False to keep them\n"
        )
        f.write("pyprep_delete_breaks = False\n")
        f.write(
            "pyprep_breaks_min_length = 20  # Minimum length of breaks in seconds to consider them as breaks\n"
        )
        f.write(
            "pyprep_t_start_after_previous = 2  # Time in seconds to start after the last event\n"
        )
        f.write(
            "pyprep_t_stop_before_next = 2  # Time in seconds to stop before the next event\n"
        )
        f.write(
            "# pyprep_custom_bad_dict: Dictionary to specify custom bad channels for each subject. The format should be {taskname :{subject:[bad_chan_list]}} for example: {'eegtask': {'001': ['TP8']}} If not specified, the bad channels will only be detected automatically.\n"
        )
        f.write("pyprep_custom_bad_dict = None\n")
        f.write("\n\n\n")
        f.write(
            "########################################################\n# Options for Icalabel \n###########################################################\n"
        )
        f.write(
            "# icalabel_bids_path: Path to the BIDS dataset for icalabel, do not change unless you want a different path from the bids files\n"
        )
        f.write("icalabel_pipeline_path = deriv_root\n")
        f.write("# icalabel_task: Task to process for icalabel, do not change\n")
        f.write("icalabel_task = task\n")
        f.write(
            "# icalabel_prob_threshold: Probability threshold to use for icalabel, default is 0.8\n"
        )
        f.write("icalabel_prob_threshold = 0.8\n")
        f.write(
            "# icalabel_labels_to_keep: List of labels to keep for icalabel, default is ['brain', 'other']\n"
        )
        f.write("icalabel_labels_to_keep = ['brain', 'other']\n")
        f.write("# icalabel_n_jobs: Number of jobs to use for icalabel, default is 1\n")
        f.write("icalabel_n_jobs = 1\n")
        f.write(
            "# icalabel_subjects: List of subjects to process for icalabel, default is same as the rest of the pipeline\n"
        )
        f.write("icalabel_subjects = subjects\n")
        f.write(
            "# icalabel_keep_mnebids_bads: Set to True to keep the bad ica already flagged in the components.tsv file (e.g. visual inspection)\n"
        )
        f.write("icalabel_keep_mnebids_bads = False\n")
        f.write("\n\n\n")
        f.write(
            "########################################################\n# Rest features extraction config\n###########################################################\n"
        )
        f.write(
            "# features_bids_path: Path to the BIDS dataset for features extraction, do not change unless you want a different path from the bids files\n"
        )
        f.write("features_bids_path = deriv_root\n")
        f.write(
            "# features_out_path: Path to the derivatives folder where the preprocessed data will be saved for features extraction\n"
        )
        f.write("features_out_path = deriv_root.replace('preprocessed', 'features')\n")
        f.write(
            "# features_task: Task to process for features extraction, do not change\n"
        )
        f.write("features_task = task\n")
        f.write(
            "# features_sourcecoords_file: Path to the source coordinates file for features extraction, default is Schaefer 2018 atlas\n"
        )
        f.write(
            "features_sourcecoords_file = os.path.join(external_root, 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv')\n"
        )
        f.write("# If 1mm file doesn't exist, try 2mm version\n")
        f.write("if not os.path.exists(features_sourcecoords_file):\n")
        f.write(
            "    features_sourcecoords_file = os.path.join(external_root, 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')\n"
        )
        f.write(
            "# features_freq_res: Frequency resolution for the PSD, default is 0.1 Hz\n"
        )
        f.write("features_freq_res = 0.1\n")
        f.write(
            "# features_freq_bands: Frequency bands to use for the PSD, default is theta, alpha, beta, gamma\n"
        )
        f.write("features_freq_bands = {\n")
        f.write("    'theta': (4, 8 - features_freq_res),\n")
        f.write("    'alpha': (8, 13 - features_freq_res),\n")
        f.write("    'beta': (13, 30),\n")
        f.write("    'gamma': (30 + features_freq_res, 80),\n")
        f.write("}\n")
        f.write(
            "# features_psd_freqmax: Maximum frequency for the PSD, default is 100 Hz\n"
        )
        f.write("features_psd_freqmax = 100\n")
        f.write(
            "# features_psd_freqmin: Minimum frequency for the PSD, default is 1 Hz\n"
        )
        f.write("features_psd_freqmin = 1\n")
        f.write(
            "# features_somato_chans: List of somatosensory channels to use for the features extraction, default is ['C3', 'C4', 'Cz']\n"
        )
        f.write("features_somato_chans = ['C3', 'C4', 'Cz']\n")
        f.write(
            "# features_subjects: List of specific subjects to compute features for, default is same as rest of pipeline)\n"
        )
        f.write("features_subjects = subjects\n")
        f.write(
            "# features_compute_sourcespace_features: Set to True to compute source space features, False to skip this step\n"
        )
        f.write("features_compute_sourcespace_features = False\n")
        f.write(
            "# features_n_jobs: Number of jobs to use for features extraction, default is 1\n"
        )
        f.write("features_n_jobs = 1\n")
        f.write(
            "# features_subjects: List of subjects to process for features extraction, default is same as the rest of the pipeline\n"
        )
        f.write("\n\n\n")
        f.write(
            "########################################################\n# ROI definitions for feature extraction\n###########################################################\n"
        )
        f.write(
            "# features_roi_channels: Dictionary defining ROI channel groups for feature extraction\n"
        )
        f.write(
            "# These ROIs are used for computing region-specific features (alpha peak, power, etc.)\n"
        )
        f.write(
            "# You can modify these definitions to match your specific EEG cap and analysis needs\n"
        )
        f.write("features_roi_channels = {\n")
        f.write(
            "    # Somatosensory Cortex (S1/S2) - pain localization and intensity coding\n"
        )
        f.write("    'somato': [\n")
        f.write("        'C3', 'C4', 'CP3', 'CP4', 'C1', 'C2', 'C5', 'C6',\n")
        f.write("        'CP1', 'CP2', 'CP5', 'CP6', 'Cz', 'CPz'\n")
        f.write("    ],\n")
        f.write("    \n")
        f.write(
            "    # Prefrontal Cortex - attention, working memory, top-down modulation\n"
        )
        f.write("    'prefrontal': [\n")
        f.write("        'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'AFz'\n")
        f.write("    ],\n")
        f.write("    \n")
        f.write("    # Frontal Cortex - cognitive control, motor planning\n")
        f.write("    'frontal': [\n")
        f.write("        'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',\n")
        f.write("        'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6'\n")
        f.write("    ],\n")
        f.write("    \n")
        f.write("    # Parietal Cortex - body representation, sensory integration\n")
        f.write("    'parietal': [\n")
        f.write("        'Pz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',\n")
        f.write("        'CP1', 'CP2', 'CP3', 'CP4', 'CPz'\n")
        f.write("    ],\n")
        f.write("    \n")
        f.write("    # Temporal Cortex - auditory processing, associative memory\n")
        f.write("    'temporal': [\n")
        f.write(
            "        'T7', 'T8', 'TP7', 'TP8', 'TP9', 'TP10', 'FT7', 'FT8', 'FT9', 'FT10'\n"
        )
        f.write("    ],\n")
        f.write("    \n")
        f.write("    # Occipital Cortex - visual perception\n")
        f.write("    'occipital': [\n")
        f.write("        'O1', 'O2', 'Oz', 'PO3', 'PO4', 'PO7', 'PO8', 'POz'\n")
        f.write("    ],\n")
        f.write("    \n")
        f.write("    # Midline - central integration (all cortical, no deep proxies)\n")
        f.write("    'midline': [\n")
        f.write("        'Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz'\n")
        f.write("    ]\n")
        f.write("}\n")
        f.write("\n\n\n")
        f.write(
            "########################################################\n# custom tfr config\n###########################################################\n"
        )
        f.write(
            "# custom_tfr_pipeline_path: Path to the preprocessed epochs, do not change unless you want a different path from the preprocessed epochs files\n"
        )
        f.write("custom_tfr_pipeline_path = deriv_root\n")
        f.write("# custom_tfr_task: Task to process for TFR, do not change\n")
        f.write("custom_tfr_task = task\n")
        f.write(
            "# custom_tfr_n_jobs: Number of jobs to use for TFR computation, default is 5\n"
        )
        f.write("custom_tfr_n_jobs = 5\n")
        f.write(
            "# custom_tfr_freqs: Frequencies to compute for TFR, default is np.arange(1, 100, 1)\n"
        )
        f.write("custom_tfr_freqs = np.arange(1, 100, 1)\n")
        f.write(
            "# custom_tfr_crop: Time interval to crop the TFR, default is None (no cropping)\n"
        )
        f.write("custom_tfr_crop = None\n")
        f.write(
            "# custom_tfr_n_cycles: Number of cycles for TFR computation, default is freqs/3.0\n"
        )
        f.write("custom_tfr_n_cycles = custom_tfr_freqs / 3.0\n")
        f.write(
            "# custom_tfr_decim: Decimation factor for TFR computation, default is 1\n"
        )
        f.write("custom_tfr_decim = 2\n")
        f.write(
            "# custom_tfr_return_itc: Whether to return inter-trial coherence, default is False\n"
        )
        f.write("custom_tfr_return_itc = False\n")
        f.write(
            "# custom_tfr_interpolate_bads: Whether to interpolate bad channels before computing the TFR, default is True\n"
        )
        f.write("custom_tfr_interpolate_bads = True\n")
        f.write(
            "# custom_tfr_average: Whether to average TFR across epochs, default is False\n"
        )
        f.write("custom_tfr_average = False\n")
        f.write(
            "# custom_tfr_return_average: Whether to return the average TFR in addition to the single trials, default is True\n"
        )
        f.write("custom_tfr_return_average = True\n")
        f.write("\n\n\n")

        if mne_bids_type == "minimal":
            f.write("\n\n\n")
            f.write(
                "########################################################\n# Rest MNE-BIDS-PIPELINE OPTIONS (MINIMAL)\n###########################################################\n"
            )
            config_text = """\
# Number of jobs
n_jobs = 10
# The task to process.
# Whether the task should be treated as resting-state data.
task_is_rest = True
# The channel types to consider.
ch_types = ["eeg"]
# Specify EOG channels to use, or create virtual EOG channels.
eog_channels = ["Fp1", "Fp2"]
# The EEG reference to use. If `average`, will use the average reference,
eeg_reference = "average"
# eeg_template_montage
eeg_template_montage = "easycap-M1"
# You can specify the seed of the random number generator (RNG).
random_state = 42
# The low-frequency cut-off in the highpass filtering step.
l_freq = 0.5
# The high-frequency cut-off in the lowpass filtering step.
h_freq = 100.0
# Specifies frequency to remove using Zapline filtering. If None, zapline will not
# be used.
zapline_fline = 60.0
# Resampling
raw_resample_sfreq = 500
# ## Epoching
# Duration of epochs in seconds.
rest_epochs_duration = 5.0  # data are segmented into 5-second epochs
# Overlap between epochs in seconds
rest_epochs_overlap = 2.5  # with a 50% overlap
epochs_tmin = 0.0
epochs_tmax = 5.0
# if `None`, no baseline correction is applied.
baseline = None
# Whether to use a spatial filter to detect and remove artifacts.
spatial_filter = "ica"
# Peak-to-peak amplitude limits to exclude epochs from ICA fitting.
ica_reject = {"eeg": 300e-6}
# The ICA algorithm to use.
ica_algorithm = "extended_infomax"  # extended infomax for mne icalabel
# ICA high pass filter for mne icalabel
ica_l_freq = 1.0
# Run source estimation or not
run_source_estimation = False
# How to handle bad epochs after ICA.
reject = "autoreject_local"
autoreject_n_interpolate = [4, 8, 16]"""
            # Split the multi-line string into a list of individual lines
            clean_text = textwrap.dedent(config_text)
            f.write(clean_text)
            f.close()
        elif mne_bids_type == "full":
            f.write(
                "########################################################\n# Rest MNE-BIDS-PIPELINE OPTIONS (Full)\n###########################################################\n"
            )
            # generate mne config file
            create_template_config(
                target_path=Path(path.replace(".py", "_temp.py")), overwrite=True
            )
            # Read the generated config file and append all lines to f
            with open(Path(path.replace(".py", "_temp.py"))) as temp_config:
                for line in temp_config:
                    f.write(line)
            f.close()
            # Delete the temporary config file
            os.remove(Path(path.replace(".py", "_temp.py")))
    logger.info(f"✅ Config file generated at {path}.")


try:
    if len(sys.argv) > 1 and sys.argv[1] == "-r":
        # Prepare to run the pipeline
        run = True
        config_file = sys.argv[2]
        mne.set_log_level("ERROR")
        plt.set_loglevel("ERROR")
        plt.switch_backend("agg")
    elif len(sys.argv) > 1 and sys.argv[1] == "-gm":
        # Generate the config file
        config_file = sys.argv[2] if len(sys.argv) > 2 else "example_config_minimal.py"
        run = False
        generate_config(config_file, mne_bids_type="minimal")
    elif len(sys.argv) > 1 and sys.argv[1] == "-gf":
        # Generate the config file
        config_file = sys.argv[2] if len(sys.argv) > 2 else "example_config_full.py"
        run = False
        generate_config(config_file, mne_bids_type="full")
    else:
        run = False
        if __name__ == "__main__":
            print(
                "ERROR! Usage: python your_script_name.py -r <config_file> to run the pipeline or -gm <config_file> to generate a minimal config file or -gf <config_file> to generate a full config file."
            )
            sys.exit(1)
except IndexError:
    run = False
    if __name__ == "__main__":
        print(
            "ERROR! Usage: python your_script_name.py -r <config_file> or -gm/-gf <config_file>."
        )
        sys.exit(1)

if run:
    """
    Run the preprocessing pipeline for all tasks specified in the config file.
    """
    #########################################################
    # Load tasks from the config file
    #########################################################
    tasks = get_config_keyval(config_file, "tasks_to_process")

    # This file path
    this_file_path = os.path.dirname(os.path.abspath(__file__))

    # Print global message
    msg = f"Welcome! 👋 The pipeline will be run sequentially for the following tasks: {', '.join(tasks)} using the data from the following BIDS folder:{get_config_keyval(config_file, 'bids_root')}."
    logger.info(msg)

    for idx, task in enumerate(tasks):
        update_config(config_file, {"task": task})
        deriv_root = get_config_keyval(config_file, "deriv_root")
        if not os.path.exists(deriv_root):
            os.makedirs(deriv_root)

        log_path = os.path.join(deriv_root, task + "_pipeline.log")

        py_command = (
            f"import sys; "
            f"sys.path.append(r'{this_file_path}'); "
            f"import {Path(__file__).stem}; "  # Dynamically import the current script
            f"{Path(__file__).stem}.run_pipeline_task(r'{task}', r'{config_file}')"
        )

        # 2. Construct the final shell command
        if get_config_keyval(config_file, "log_type") == "file":
            cmd = f'python -c "{py_command}" > "{log_path}" 2>&1'
            msg = f"👍 Running pipeline for task {idx+1} out of {len(tasks)}: {task} and logging to {log_path}"
            logger.info(msg)
        else:
            cmd = f'python -c "{py_command}"'
            print(
                f"👍 Running pipeline for task {idx+1} out of {len(tasks)}: {task} and logging to console"
            )

        exit_code = os.system(cmd)
        if exit_code != 0:
            logger.error(
                f"❌ Pipeline failed for task {task}. Check the log file for details."
            )
            sys.exit(1)
        else:
            logger.info(f"✅ Pipeline completed successfully for task {task}.")

#########################################
# Plotting functions
#########################################


def plot_tfr_nice_spectro(
    tfr: AverageTFR,
    chans: List[str],
    cmap: str = "RdBu_r",
    time_range: Tuple[Optional[float], Optional[float]] = (None, None),
    freq_range: Optional[Tuple[float, float]] = None,
    vlim: Tuple[float, float] = (-3, 3),
    figsize: Tuple[float, float] = (2, 1),
    nameout: str = "",
    title: str = "",
    cbar: bool = True,
    fontsize: int = 8,
    save_path: Optional[str] = None,
    cbar_label: str = "Z-score",
    y_ticks: Optional[List[float]] = None,
    x_ticks: Optional[List[float]] = None,
    hide_xlabel: bool = False,
    hide_ylabel: bool = False,
    remove_xticks: bool = False,
    remove_yticks: bool = False,
    remove_xticklabels: bool = False,
    remove_yticklabels: bool = False,
    extension: str = "svg",
    dpi: int = 1200,
    bbox_inches: str = "tight",
    transparent: bool = True,
    show=False,
):
    """Plot a single channel time-frequency representation (TFR) as a spectrogram."""
    # Create the figure and axes objects
    fig, ax = plt.subplots(figsize=figsize)

    # Use the MNE plot function to draw the main spectrogram
    tfr.plot(
        chans,
        baseline=None,
        show=False,
        vlim=vlim,
        tmin=time_range[0],
        tmax=time_range[1],
        axes=ax,
        cmap=cmap,
        title=None,
        colorbar=cbar,
        combine="mean",
    )

    # --- Customize Plot Appearance ---
    ax.tick_params(axis="both", which="major", labelsize=fontsize - 2)
    ax.set_ylabel("Frequency (Hz)", fontsize=fontsize)
    ax.set_xlabel("Time (s)", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    if time_range[0] is not None or time_range[1] is not None:
        ax.set_xlim(time_range)
    if freq_range is not None:
        ax.set_ylim(freq_range)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)

    if cbar and len(fig.axes) > 1:
        cbar_ax = fig.axes[1]
        cbar_ax.set_ylabel(cbar_label, fontsize=fontsize)
        cbar_ax.tick_params(labelsize=fontsize - 2)

    # --- Hide or Remove Plot Elements ---
    if hide_ylabel:
        ax.set_ylabel("")
    if remove_yticks:
        ax.set_yticks([])
    if hide_xlabel:
        ax.set_xlabel("")
    if remove_xticks:
        ax.set_xticks([])
    if remove_xticklabels:
        ax.set_xticklabels([])
    if remove_yticklabels:
        ax.set_yticklabels([])

    # --- Save Figure ---
    if save_path is not None:
        full_path = os.path.join(
            save_path,
            f"{nameout}_spectro.{extension}",
        )
        fig.savefig(
            full_path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=transparent,
        )
    if not show:
        plt.close(fig)


def plot_tfr_nice_topo(
    tfr: AverageTFR,
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
    title: str = "",
    cmap: str = "RdBu_r",
    vlim: Tuple[float, float] = (-1.5, 1.5),
    figsize: Tuple[float, float] = (1.5, 1.5),
    nameout: str = "",
    cbar: bool = True,
    fontsize: int = 10,
    cbar_label: str = "Z-score",
    save_path: Optional[str] = None,
    extension: str = "png",
    dpi: int = 800,
    bbox_inches: str = "tight",
    transparent: bool = True,
    show: bool = False,
):
    """Plot a time-frequency representation (TFR) as a topographic map."""
    if mne is None:
        raise ImportError("MNE-Python must be installed to use this function.")

    avg_power_topo = (
        tfr.copy()
        .crop(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
        .get_data()
        .mean(axis=(1, 2))
    )

    fig, ax = plt.subplots(figsize=figsize)
    im, _ = mne.viz.plot_topomap(
        avg_power_topo,
        tfr.info,
        vlim=vlim,
        cmap=cmap,
        contours=False,
        axes=ax,
        show=False,
    )

    ax.set_title(title, fontsize=fontsize)

    if cbar:
        cax = fig.add_axes(
            [
                ax.get_position().x1 + 0.01,
                ax.get_position().y0,
                0.04,
                ax.get_position().height,
            ]
        )
        clb = fig.colorbar(im, cax=cax)
        clb.set_label(cbar_label, fontsize=fontsize - 2)
        clb.ax.tick_params(labelsize=fontsize - 3)

    if save_path is not None:
        full_path = os.path.join(
            save_path,
            f"{nameout}_topo.{extension}",
        )
        fig.savefig(
            full_path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=transparent,
        )

    if not show:
        plt.close(fig)


def rm_ttest_tfr(
    tfr_dict1,
    tfr_dict2,
    freq_range=None,
    time_range=None,
    channels=None,
    n_permutations=1000,
    clusters=None,
    tail=0,
    n_jobs=1,
    seed=None,
    decimate=None,
    threshold=None,
    adjacency_name="easycap-M1",
    return_stats_as_evoked=False,
    alpha=0.05,
):
    if mne is None:
        raise ImportError("MNE-Python must be installed to use this function.")

    if set(tfr_dict1.keys()) != set(tfr_dict2.keys()):
        raise ValueError("Both TFR dictionaries must have the same keys.")

    for key in tfr_dict1.keys():
        tfr_dict1[key].pick_types(eeg=True)
        if tfr_dict2 is not None:
            tfr_dict2[key].pick_types(eeg=True)

    if freq_range is not None:
        for tfr in tfr_dict1.values():
            tfr.crop(fmin=freq_range[0], fmax=freq_range[1])
        if tfr_dict2 is not None:
            for tfr in tfr_dict2.values():
                tfr.crop(fmin=freq_range[0], fmax=freq_range[1])

    if time_range is not None:
        for tfr in tfr_dict1.values():
            tfr.crop(tmin=time_range[0], tmax=time_range[1])
        if tfr_dict2 is not None:
            for tfr in tfr_dict2.values():
                tfr.crop(tmin=time_range[0], tmax=time_range[1])

    if channels is not None:
        for tfr in tfr_dict1.values():
            tfr.pick(channels)
        if tfr_dict2 is not None:
            for tfr in tfr_dict2.values():
                tfr.pick(channels)

    if decimate is not None:
        for tfr in tfr_dict1.values():
            tfr.decimate(decimate, verbose=False)
        if tfr_dict2 is not None:
            for tfr in tfr_dict2.values():
                tfr.decimate(decimate, verbose=False)

    data1 = np.array([tfr.data for tfr in tfr_dict1.values()])
    if tfr_dict2 is not None:
        data2 = np.array([tfr.data for tfr in tfr_dict2.values()])
        data_diff = data1 - data2
    else:
        data_diff = data1

    tfr = list(tfr_dict1.values())[0]
    assert np.argwhere(np.asarray(data_diff.shape) == len(tfr_dict1))[0][0] == 0
    assert (
        np.argwhere(np.asarray(data_diff.shape) == len(tfr.info["ch_names"]))[0][0] == 1
    )
    assert np.argwhere(np.asarray(data_diff.shape) == len(tfr.freqs))[0][0] == 2
    assert np.argwhere(np.asarray(data_diff.shape) == len(tfr.times))[0][0] == 3

    if clusters is None:
        X = data_diff.reshape(data_diff.shape[0], -1)
        t_stat, p_val = mne.stats.permutation_t_test(
            X, n_permutations=n_permutations, tail=tail, n_jobs=n_jobs, seed=seed
        )
        t_stat = t_stat.reshape(data_diff.shape[1:])
        p_val = p_val.reshape(data_diff.shape[1:])
    else:
        if adjacency_name is not None:
            sensor_adjacency, _ = mne.channels.read_ch_adjacency(adjacency_name)
        else:
            sensor_adjacency, _ = mne.channels.find_ch_adjacency(
                tfr.info, ch_type="eeg", verbose=False
            )

        adjacency = mne.stats.combine_adjacency(
            sensor_adjacency, len(tfr.freqs), len(tfr.times)
        )

        if threshold is None:
            df = data_diff.shape[0] - 1
            if tail == 0:
                t_thresh = scipy.stats.distributions.t.ppf(1 - alpha / 2, df=df)
            elif tail == 1:
                t_thresh = scipy.stats.distributions.t.ppf(1 - alpha, df=df)
        elif threshold == "TFCE":
            t_thresh = dict(start=0, step=0.2)

        t_stat, clusters, cluster_p_values, H0 = (
            mne.stats.spatio_temporal_cluster_1samp_test(
                data_diff,
                n_permutations=n_permutations,
                tail=tail,
                n_jobs=n_jobs,
                seed=seed,
                adjacency=adjacency,
                threshold=t_thresh,
            )
        )

        t_stat = t_stat.reshape(data_diff.shape[1:])
        p_val = np.ones_like(t_stat) * np.nan
        for cl, p in zip(clusters, cluster_p_values):
            p_val[cl] = p

    if return_stats_as_evoked:
        cp = tfr.copy()
        cp.data = t_stat[np.newaxis, :, :]
        t_stat = cp.copy()
        cp_p = tfr.copy()
        cp_p.data = p_val[np.newaxis, :, :]
        p_val = cp_p.copy()

    return t_stat, p_val, list(tfr_dict1.values())[0].info
