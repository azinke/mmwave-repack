# MMWave Repack

This is a tool for reconstruction MIMO frames based on the recordings from each
radar device on the Cascased MMWCAS-RF-EVM module.

After a successful recording, separate recording files are created for each of
the four radar devices as follow:

```txt
├── master_0000_data.bin
├── master_0000_idx.bin
├── slave1_0000_data.bin
├── slave1_0000_idx.bin
├── slave2_0000_data.bin
├── slave2_0000_idx.bin
├── slave3_0000_data.bin
└── slave3_0000_idx.bin
```

This tool thus merges the recording from the radar devices to and split them into
MIMO frames. The order of the data in each MIMO frame respects the antenna layout
on the MMWCAS-RF-EVM board. Hence, the chirp recordings are saved in MIMO frames
for each radar device in the following order

```txt
[DEV 4] [DEV 1] [DEV 3] [DEV 2]

With

[DEV 1]: Master
[DEV 2]: Slave 3
[DEV 3]: Slave 2
[DEV 4]: Slave 1
```

**NOTE**: In this tool,

- it's assumed that all the RX antenna are enabled as well
as all the TX antenna. For custom layout, the script must be updated.
- it's also assumed that recordings are in complex format. As so, for each samples,
it's assumed that I and Q signals has been recorded.


## Usage

A quick overview of the currently supported CLI options are presented below.

```txt
usage: repack.py [-h] [-v] [-ns NUMBER_SAMPLES] [-nc NUMBER_CHIRPS] [-o OUTPUT_DIR]
                 [-i INPUT_DIR]

MMWAVECAS-RF-EVM board recordings post-processing routine. Repack the recordings into
MIMO frames

options:
  -h, --help            show this help message and exit
  -v, --version         Print software version and information about the dataset.
  -ns NUMBER_SAMPLES, --number-samples NUMBER_SAMPLES
                        Number of samples per chirp
  -nc NUMBER_CHIRPS, --number-chirps NUMBER_CHIRPS
                        Number of chirps per frame
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory for storing the mimo frames
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Inpout directory containing the recordings

```

The default values of these options are:

```bash
-ns 256 -nc 16 -o output
```

### Examples

1. Only providing the recording directory

```bash
python repack.py -i <path-to-recording-folder>
```

This means that default options would be used for any other option.
A folder named `output` will automatically be created within the recording folder to hold
the generated MIMO frames. 

2. Defining input and output directories

```bash
python repack.py -i <path-to-recording-folder> -o <path-to-output-directory>
```

If the output directory does not already exist, it will be created. Other options would
keep their default values and MIMO frames will be saved in the specified output directory.

2. Defining the number of samples per chirp or chrip loop

```bash
python repack.py -ns <nb-samples-per-chrip> -nc <nb-chrip-loop> -i <path-to-recording-folder>
```


## Reading and parsing MIMO frames

After merging and repacking the data, MIMO frames are generated and saved
as a sequence of `int16` integers in binary files (`.bin`). Such files can be read
as shown the code snippet below:

```python
import numpy as np

ntx: int = 12       # Number of TX antenna enalbled (3 TX antenna per radar device)
nrx: int = 16       # Number of RX antenna enabled (4 RX antenna per radar device)
nchirp: int = 16    # Number of chrip loop
ns: int = 256       # Number of samples per chirp

frame = np.fromfile("frame1.bin", dtype=np.int16, count=-1)

# "2" axis count for the I and Q signals recorded
frame = frame.reshape(ntx, nrx, nchirp, ns, 2)
# Frame fully reconstructed
frame = frame[:, :, :, :, 0] + 1j * frame[:, :, :, :, 1]

```
