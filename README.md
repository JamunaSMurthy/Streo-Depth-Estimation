# Stereo depth estimation
Simple implementation of stereo depth estimation.

## What is stereo depth?

## What you will do?

We have already trained a simple neural network for stereo depth estimation. That is, given two left and right images, the network estimates a disparity map.

You will first run this code and see the results.

Before that, you will need to setup a virtual environment to run the code.

## How to Run

To run this repository, please follow the steps below:

1. Clone the repository:

    ```bash
    git clone https://github.com/jinsuyoo/simple-stereo.git
    ```

2. Navigate to the project directory:

    ```bash
    cd simple-stereo
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the stereo depth estimation script:

    ```bash
    python test.py
    ```

5. The script will output the stereo depth estimation results.

That's it! You have successfully run the simple-stereo repository. Feel free to explore the code and make any modifications as needed.


# Acknowledgment

Implementation is based on [PSMNet](https://github.com/JiaRenChang/PSMNet) and [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo).