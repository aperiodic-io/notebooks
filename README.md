# Aperiodic Notebooks

Interactive [marimo](https://marimo.io) notebooks for exploring Aperiodic market data APIs.

These notebooks are used in the [Aperiodic Playground](https://aperiodic.io/home/playground) and can also be run standalone on your machine.

## Installation

```bash
pip install marimo aperiodic[pandas] pyarrow matplotlib
```

Or install from this repo (includes dev tooling):

```bash
git clone https://github.com/aperiodic-io/notebooks.git
cd notebooks
pip install -e ".[quality]"
```

## Usage

### Interactive editor

Open a notebook in the marimo editor:

```bash
marimo edit notebooks/getting-started.py
```

### Run as a script

Execute a notebook as a standard Python script:

```bash
python notebooks/getting-started.py
```

### Run as a web app

Serve a notebook as a read-only web application:

```bash
marimo run notebooks/getting-started.py
```

## Available Notebooks

| Notebook | Description |
|----------|-------------|
| `getting-started.py` | OHLCV data fetching, charting, and basic flow metrics |
| `derivatives.py` | Funding rates, open interest, and basis analysis |
| `order-flow.py` | Taker flow, buy/sell imbalances, and cumulative delta |

## Requirements

- Python 3.11+
- An [Aperiodic API key](https://aperiodic.io)

## License

MIT
