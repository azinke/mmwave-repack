"""Format MMWCAS-RF-EVM & MMWCAS-DSP-EVM kit Recordings.

    @author: AMOUSSOU Z. Kenneth
    @date: 13-08-2022
"""
from typing import Optional
from datetime import timedelta, datetime
import os
import glob
import argparse
import sys
import numpy as np

__VERSION__: str = "0.1"
__COPYRIGHT__: str = "Copyright (C) 2022, RWU-RADAR Project"


def getInfo(idx_file: str) -> tuple[int, int]:
    """Get information about the recordings.

    The "*_idx.bin" files along the sample files gather usefule
    information aout the dataset.

    The structure of the "*_idx.bin" file is as follow:

    ---------------------------------------------------------------------------
         File header in *_idx.bin:
             struct Info
             {
                 uint32_t tag;
                 uint32_t version;
                 uint32_t flags;
                 uint32_t numIdx;       // number of frames
                 uint64_t dataFileSize; // total data size written into file
             };

         Index for every frame from each radar:
             struct BuffIdx
             {
                 uint16_t tag;
                 uint16_t version; /*same as Info.version*/
                 uint32_t flags;
                 uint16_t width;
                 uint16_t height;

                 /*
                  * For image data, this is pitch. For raw data, this is
                  * size in bytes per metadata plane
                  */
                 uint32_t pitchOrMetaSize[4];

                /*
                 * Total size in bytes of the data in the buffer
                 * (sum of all planes)
                */
                 uint32_t size;
                 uint64_t timestamp; // timestamp in ns
                 uint64_t offset;
             };

        Source: Example matlab script provided by Texas Instrument
    ---------------------------------------------------------------------------

    Arguemnt:
        idx_file: Path to an index file from any of the cascaded chip

    Return:
        Tuple containing respectively the number of valid frames recorded
        and the size of the data file
    """
    # Data type based on the structure of the file header
    dt = np.dtype([
        ("tag", np.uint32),
        ("version", np.uint32),
        ("flags", np.uint32),
        ("numIdx", np.uint32),
        ("size", np.uint64),
    ])
    header = np.fromfile(idx_file, dtype=dt, count=1)[0]

    dt = np.dtype([
        ("tag", np.uint16),
        ("version", np.uint16),
        ("flags", np.uint32),
        ("width", np.uint16),
        ("height", np.uint16),

        ("_meta0", np.uint32),
        ("_meta1", np.uint32),
        ("_meta2", np.uint32),
        ("_meta3", np.uint32),

        ("size", np.uint32),
        ("timestamp", np.uint64),
        ("offset", np.uint64),
    ])

    data = np.fromfile(idx_file, dtype=dt, count=-1, offset=24)
    timestamps = np.array([
        (datetime.now() + timedelta(seconds=log[-2] * 1e-9)).timestamp()
        for log in data
    ])

    return header[3], header[4], timestamps


def load(inputdir: str, device: str) -> Optional[dict[str, list[str]]]:
    """Load the recordings of the radar chip provided in argument.

    Arguments:
        inputdir: Input directory to read the recordings from
        device: Name of the device

    Return:
        Dictionary containing the data and index files
    """
    # Collection of the recordings data file
    # They all have the "*.bin" ending
    recordings: dict[str, list[str]] = {
        "data": glob.glob(f"{inputdir}{os.sep}{device}*data.bin"),
        "idx": glob.glob(f"{inputdir}{os.sep}{device}*idx.bin")
    }
    recordings["data"].sort()
    recordings["idx"].sort()

    if (len(recordings["data"]) == 0) or (recordings["idx"] == 0):
        print(f"[ERROR]: No file found for device '{device}'")
        return None
    elif len(recordings["data"]) != len(recordings["idx"]):
        print(
            f"[ERROR]: Missing {device} data or index file!\n"
            "Please check your recordings!"
            "\nYou must have the same number of "
            "'*data.bin' and '*.idx.bin' files."
        )
        return None
    return recordings


def toframe(
        mf: str, sf0: str, sf1: str, sf2: str,
        ns: int, nc: int, nf: int,
        output: str = ".",
        start_idx: int = 0) -> int:
    """Re-Format the raw radar ADC recording.

    The raw recording from each device is merge together to create
    separate recording frames corresponding to the MIMO configuration.

    Arguments:
        mf: Path to the recording file of the master device
        sf0: Path to the recording file of the first slave device
        sf1: Path to the recording file of the second slave device
        sf2: Path to the recording file of the third slave device

        ns: Number of ADC samples per chirp
        nc: Number of chrips per frame
        nf: Number of frames to generate

        output: Path to the output folder where the frame files would
                be written

        start_idx: Index to start numbering the generated files from.

    Return:
        The index number of the last frame generated

    Note:
        Considering the AWR mmwave radar kit from Texas Instrument used,
        The following information can be infered:
            - Number of cascaded radar chips: 4
            - Number of RX antenna per chip: 4
            - Number of TX antenna per chip: 3
            - Number of signal measured per ADC samples: 2
                - In-phase signal (I)
                - Quadrature signal (Q)
    """
    # Number of waveform measured
    # 2-bytes (signed integer) for I (In-phase signal)
    # 2.bytes (signed integer) for Q (Quadrature signal)
    nwave: int = 2

    # Number of cascaded radar chips
    nchip: int = 4

    # Number of TX antenna
    ntx: int = 3

    # Number of RX antenna per chip
    nrx: int = 4

    # Number of frame to skip at the beginning of the recording
    nf_skip: int = 0

    # Index used to number frames
    fk: int = start_idx

    for fidx in range(nf_skip, nf):
        # Number of items to read (here items are 16-bit integer values)
        nitems: int = nwave * ns * nc * nrx * ntx * nchip
        # Offet to read the bytes of a given frame
        # The multiplication by "2" is to count for the size of 16-bit integers
        offset: int = fidx * nitems * 2

        dev1 = np.fromfile(mf, dtype=np.int16, count=nitems, offset=offset)
        dev2 = np.fromfile(sf0, dtype=np.int16, count=nitems, offset=offset)
        dev3 = np.fromfile(sf1, dtype=np.int16, count=nitems, offset=offset)
        dev4 = np.fromfile(sf2, dtype=np.int16, count=nitems, offset=offset)

        dev1 = dev1.reshape(nc, ntx * nchip, ns, nrx, 2)
        dev2 = dev2.reshape(nc, ntx * nchip, ns, nrx, 2)
        dev3 = dev3.reshape(nc, ntx * nchip, ns, nrx, 2)
        dev4 = dev4.reshape(nc, ntx * nchip, ns, nrx, 2)

        dev1 = np.transpose(dev1, (1, 3, 0, 2, 4))
        dev2 = np.transpose(dev2, (1, 3, 0, 2, 4))
        dev3 = np.transpose(dev3, (1, 3, 0, 2, 4))
        dev4 = np.transpose(dev4, (1, 3, 0, 2, 4))

        frame = np.zeros((nchip * ntx, nrx * nchip, nc, ns, 2))
        frame[:, 0:4, :, :] = dev4
        frame[:, 4:8, :, :] = dev1
        frame[:, 8:12, :, :] = dev3
        frame[:, 12:16, :, :] = dev2

        # Name for saving the frame
        fname: str = f"frame_{fk}.bin"
        fpath: str = os.path.join(output, fname)
        frame.astype(np.int16).tofile(fpath)
        print(f"Frame {fk} written!", end="\r")
        fk += 1
    # Return the index of the last frame generated
    return fk - 1


if __name__ == "__main__":
    # Output directory that would hold the formatted data per frame
    OUTPUT_DIR: str = "output"

    # Number of samples
    NS: int = 256

    # Number of chirps
    NC: int = 16

    parser = argparse.ArgumentParser(
        prog="repack.py",
        description="MMWAVECAS-RF-EVM board recordings post-processing routine. "
                    "Repack the recordings into MIMO frames"
    )
    parser.add_argument(
        "-v", "--version",
        help="Print software version and information about the dataset.",
        action="store_true"
    )
    parser.add_argument(
        "-ns", "--number-samples",
        help="Number of samples per chirp ",
        type=int,
        default=NS,
    )
    parser.add_argument(
        "-nc", "--number-chirps",
        help="Number of chirp loops per frame ",
        type=int,
        default=NC,
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory for storing the mimo frames",
        type=str,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "-i", "--input-dir",
        help="Input directory containing the recordings",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.version:
        print(f"mmwave-repack version {__VERSION__}, {__COPYRIGHT__}")
        sys.exit(0)

    if args.input_dir is None:
        print("[ERROR]: Missing input directory to read recordings")
        sys.exit(1)

    # The output directory will be created inside the data directory by default
    if (args.input_dir is not None) and (args.output_dir == OUTPUT_DIR):
        args.output_dir = os.path.join(args.input_dir, OUTPUT_DIR)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Load devices recording file paths
    master: dict[str, list[str]] = load(args.input_dir, "master")
    slave1: dict[str, list[str]] = load(args.input_dir, "slave1")
    slave2: dict[str, list[str]] = load(args.input_dir, "slave2")
    slave3: dict[str, list[str]] = load(args.input_dir, "slave3")

    assert master != None, "Error with master data files"
    assert slave1 != None, "Error with slave1 data files"
    assert slave2 != None, "Error with slave2 data files"
    assert slave3 != None, "Error with slave3 data files"

    # Integrity status of the recordings
    # Check if the number of files generated for each device is
    # identical
    status: bool = True

    status = status and (len(master["data"]) == len(slave1["data"]))
    status = status and (len(master["data"]) == len(slave2["data"]))
    status = status and (len(master["data"]) == len(slave3["data"]))

    if not status:
        print("[ERROR]: Missing recording for cascade MIMO configuration")
        sys.exit(1)

    size: int = len(master["data"])

    # Number of frames recorded
    nf: int = 0

    # Number of frames generated from the last recording batch
    previous_nf: int = 0

    timestamps = np.array([])

    for idx in range(size):
        # Master file
        mf: str = master["data"][idx]
        mf_idx: str = master["idx"][idx]

        # Slave data files
        sf1: str = slave1["data"][idx]
        sf2: str = slave2["data"][idx]
        sf3: str = slave3["data"][idx]

        nf, _, timelogs = getInfo(mf_idx)

        # Skip if the number of valid frame is 0
        if not nf:
            continue

        timestamps = np.append(timestamps, timelogs)

        previous_nf = toframe(
            mf, sf1, sf2, sf3, # Input data files
            args.number_samples,
            args.number_chirps,
            nf,
            args.output_dir,
            start_idx=previous_nf + 1
        )
    print(f"[SUCCESS]: {previous_nf:04} frames written!")
    # Save all the timestamps in a single file
    timestamps.tofile(os.path.join(args.output_dir, "timestamps.txt"), "\n")

    print(f"[SUCCESS]: {previous_nf:04d} MIMO frames successfully generated!")
