# Option Pricing with Stochastic Volatility — M2R Group 51

## Project Structure

```text
Option_Pricing_with_Stochastic_Volatility_Model_M2R_Group_51/
├── pyproject.toml          # Dependency config — do not modify
├── test.py                 # Run this to verify your setup
├── src/
│   └── pricing/
│       ├── __init__.py
│       └── models.py       # Pricing models (BSM, CRR, MC, FD)
├── notebooks/              # Put your notebooks here
└── .gitignore
```

---

## Setup Guide

> Open the project root folder in VS Code before starting.

### Step 1 — Create a virtual environment

Open a terminal in VS Code via **Terminal > New Terminal**, then run:

**Mac / Linux**
```bash
python3 -m venv .venv
```

**Windows**
```bash
python -m venv .venv
```

### Step 2 — Activate and install

**Mac / Linux**
```bash
source .venv/bin/activate
pip install -e .
```

**Windows**
```bash
.venv\Scripts\activate
pip install -e .
```

Your terminal prompt should now show `(.venv)`. If it does not, the environment is not active — do not proceed.

> `pip install -e .` installs the `pricing` package in editable mode, meaning any changes to `models.py` are reflected immediately without reinstalling.

### Step 3 — Select the VS Code interpreter

VS Code may default to your system Python or Anaconda. You must manually point it to the virtual environment.

1. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows)
2. Type **Python: Select Interpreter** and press Enter
3. Select the entry showing `./.venv/bin/python` (Mac/Linux) or `.\.venv\Scripts\python.exe` (Windows)

> **Do not select Anaconda or the system default.**

### Step 4 — Verify your setup

```bash
python test.py
```

If everything is configured correctly, you will see the option prices printed in the terminal.

---

## Usage

```python
from pricing.models import bsm_european, binomial_crr_european

price = bsm_european(S=100, K=100, T=1, r=0.05, sigma=0.2)
print(price)
```

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: No module named 'pricing'` | Wrong interpreter or inactive environment | Check for `(.venv)` in terminal; repeat Step 3 |
| `pip: command not found` | Python not installed or not on PATH | Reinstall Python from [python.org](https://www.python.org) |
| Want a clean reinstall | Corrupted environment | Delete `.venv` folder and repeat Steps 1–3 |

**Before asking for help, confirm you have:**
- [ ] Opened the correct root folder in VS Code
- [ ] Seen `(.venv)` appear in your terminal prompt
- [ ] Run `pip install -e .`
- [ ] Selected `.venv` as the interpreter in VS Code