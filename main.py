"""Format Recordings."""
from typing import Optional
import os
import glob
import argparse
import sys
import numpy as np


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
                 uint64_t timestamp;
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
    return header[3], header[4]


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
    # 2-bytes (signed integer) for I (In-phase signal
    # 2.bytes (signed integer) for Q (Quadrature signal)
    nwave: int = 2

    # Number of cascaded radar chips
    nchip: int = 4

    # Number of TX antenna
    ntx: int = 3

    # Number of RX antenna per chip
    nrx: int = 4

    # Number of frame to skip at the beginning of the recording
    nf_skip: int = 2

    # Index of the last frame generated
    last_idx: int = 0

    for fidx in range(nf_skip, nf + nf_skip):
        # Number of bytes to read
        nbytes: int = nwave * ns * nc * ntx * nrx * nchip
        # Offet to read the bytes of a given frame
        offset: int = fidx * nbytes

        dev1 = np.fromfile(mf, dtype=np.int16, count=nbytes, offset=offset)
        dev2 = np.fromfile(sf0, dtype=np.int16, count=nbytes, offset=offset)
        dev3 = np.fromfile(sf1, dtype=np.int16, count=nbytes, offset=offset)
        dev4 = np.fromfile(sf2, dtype=np.int16, count=nbytes, offset=offset)

        dev1 = dev1.reshape(nchip * ntx, nrx, nc, ns, 2)
        dev2 = dev2.reshape(nchip * ntx, nrx, nc, ns, 2)
        dev3 = dev3.reshape(nchip * ntx, nrx, nc, ns, 2)
        dev4 = dev4.reshape(nchip * ntx, nrx, nc, ns, 2)

        frame = np.zeros((nchip * ntx, nrx * nchip, nc, ns, 2))

        for tidx in range(nchip * ntx):
            for ridx in range(nrx):
                frame[tidx, :, :, :, :] = np.vstack((
                    dev1[tidx, :, :, :, :],
                    dev2[tidx, :, :, :, :],
                    dev3[tidx, :, :, :, :],
                    dev4[tidx, :, :, :, :],
                ))
        # Name for saving the frame
        fname: str = f"frame_{fidx -  nf_skip + start_idx}.bin"
        fpath: str = os.path.join(output, fname)
        frame.astype(np.int16).tofile(fpath)
        print(f"Frame {fidx -  nf_skip + start_idx} written!")
        last_idx += 1
    print()
    return last_idx


if __name__ == "__main__":
    # Output directory that would hold the formatted data per frame
    OUTPUT_DIR: str = "output"

    # Input directory containing the recordings from mmwave studio
    INPUT_DIR: str = "."

    # Number of samples
    NS: int = 256

    # Number of chirps
    NC: int = 64

    parser = argparse.ArgumentParser(
        prog="MMWAVECAS-RF-EVM board recordings post-processing routine",
        description="Repack the recordings into MIMO frames"
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
        help="Number of chirps per frame ",
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
        help="Inpout directory containing the recordings",
        type=str,
        default=INPUT_DIR,
    )

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

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
        print("[ERROR]: Missing recording for cascase mimo configuration")
        sys.exit(1)

    size: int = len(master["data"])

    # Number of frames recorded
    nf: int = 0

    # Number of frames generated from the last recording batch
    previous_nf: int = 0

    for idx in range(size):
        # Master file
        mf: str = master["data"][idx]
        mf_idx: str = master["idx"][idx]

        # Slave data files
        sf1: str = slave1["data"][idx]
        sf2: str = slave1["data"][idx]
        sf3: str = slave1["data"][idx]

        nf, _ = getInfo(mf_idx)

        toframe(
            mf, sf1, sf2, sf3, # Input data files
            args.number_samples,
            args.number_chirps,
            nf,
            args.output_dir,
            start_idx=previous_nf + 1
        )
        previous_nf += nf

    print(f"[SUCCESS]: {previous_nf:04d} MIMO frames successfully generated!")
