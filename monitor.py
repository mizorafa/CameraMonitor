#!/usr/bin/env python

"""
 ---- 13th June, 2023 ----
Developed by S. Abe, using a ClusCo log reader that Seiya developed.
usage: 
put this script with "sclusco.py", change "path_clog" in main(), and run "python monitor.py".
future development:
 - more smooth plot update: maybe by "matpltlib.animation" or "CameraDisplay.update".
 - a unified configutation
 - a smarter normalization
"""

import glob
import datetime
import numpy as np
from matplotlib import pyplot as plt
from sclusco import ClusCoLog


def fill_monitor(
    fig, clusco_state,
):

    # config
    # ToDo: relocate to main, automatic ncosl/nrows
    ncols = 3
    nrows = 2
    idx = -1 # -1 or np.random.randint(int)
    
    # subplots
    axes = fig.subplots(nrows=nrows, ncols=ncols)
    displays = [
        # High Voltage
        clusco_state.high_voltage.show_snapshot(
            ax=axes[0,0], idx=idx, 
            # norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=True), # ToDo: not work?
        ),
        # Anode Current
        clusco_state.anode_current.show_snapshot(ax=axes[0,1], idx=idx),
        # Amplifier Temperature
        clusco_state.amp_temp.show_snapshot(ax=axes[0,2], idx=idx),
        # L0 Rate
        clusco_state.l0_rate.show_snapshot(ax=axes[1,0], idx=idx),
        # L1 Rate
        clusco_state.l1_rate.show_snapshot(ax=axes[1,1], idx=idx),
        # Camera Rate
        clusco_state.camera_rate.show_snapshot(ax=axes[1,2], idx=idx),
    ]
    # adjustment
    fig.tight_layout()

    return fig, axes, displays


def monitor(
    fig, path_clog, timestamp,
):
    
    # read the ClusCo log
    filenames = sorted(glob.glob(path_clog))
    filename = filenames[-1]
    log_clusco = ClusCoLog(filename=filename)
    # read the status
    clusco_states = log_clusco.states
    fig.suptitle(
        f"{timestamp.strftime('%c')}"
        "\n"
        f"{filename}"
    )
    # make monitor window
    fig, axes, displays = fill_monitor(fig, clusco_states)
    print(timestamp)

    return fig, axes, displays



def main():

    max_duration = datetime.timedelta(minutes=30)
    frame_time = 5.0 # seconds
    path_clog = "../Camera/data/*.txt"

    # start/end time
    start = datetime.datetime.now()
    expected_end = start + max_duration
    # figure object
    fig = plt.figure(figsize=(13,8))
    fig.show()
    # initial plot
    fig, axes, displays = monitor(fig, path_clog, datetime.datetime.now())
    plt.pause(frame_time)

    timestamp = datetime.datetime.now()
    while timestamp < expected_end:

        # reset plot
        fig.clf()
        # read the latest file
        fig, axes, displays = monitor(fig, path_clog, timestamp)
        plt.pause(frame_time)
        # time stamp update
        timestamp = datetime.datetime.now()



if __name__ == '__main__':
    main()