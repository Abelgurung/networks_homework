# CS 565 Computer Networks Assignment 1

This project contains scripts for network analysis, utilizing a pinger written in Rust and Python scripts for data analysis and plotting.

**Note:** This project was created with the help of LLMs.

## Prerequisites

Before running the project, ensure you have the following installed:

- [Rust](https://www.rust-lang.org/tools/install) (including `cargo`)
- [Python 3](https://www.python.org/downloads/)

## Quick Start

The project includes a `run.sh` script that automates the build and execution process. To run the entire project in one shot:

```bash
./run.sh
```

## What does `run.sh` do?

The `run.sh` script performs the following steps:

1.  **Builds the Pinger**: Compiles the Rust project located in the `pinger/` directory.
2.  **Collects Data**: Runs the compiled pinger using targets from `list.txt` and saves the output to `data.txt`.
3.  **Sets up Python Environment**: Creates a virtual environment (`.venv`) if one doesn't exist and activates it.
4.  **Installs Dependencies**: Installs required Python packages (`pandas`, `requests`, `matplotlib`) from `requirements.txt`.
5.  **Runs Analysis**: Executes `q1.py` and `q2.py` to analyze the data and generate results.

## Manual Setup

If you cannot run the shell script or prefer to run steps manually:

1.  **Build and Run Pinger (Rust)**
    ```bash
    cd pinger
    cargo build --release
    cd ..
    ./pinger/target/release/pinger list.txt > data.txt
    ```

2.  **Install Python Dependencies**
    ```bash
    python3 -m venv .venv
    # Activate venv:
    # On macOS/Linux: source .venv/bin/activate
    # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Run Analysis Scripts**
    ```bash
    python3 q1.py data.txt
    python3 q2.py list.txt
    ```
