"""Generate clean Kaggle notebook for RQ3 ablations."""
import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# BETA RQ3 Hyperparameter Ablations (Tables 12 & 13)\n",
                "\n",
                "This notebook evaluates the 7 hyperparameter configurations for RQ3:\n",
                "- **Table 12** (Pheromone update rules): Exponential (default), Rank, Uniform\n",
                "- **Table 13** (Ants count M): 5, 10, 15, 20 (default), 25\n",
                "\n",
                "**Consistency Guarantee**: The default configuration (K=1, H=3, update='exponential', M=20) is shared between Table 12 Exponential and Table 13 M=20, guaranteeing identical numbers.\n",
                "\n",
                "### Parallel Execution Strategy across 2 Kaggle Accounts (10 commits total):\n",
                "Make 10 copies of this notebook (or run with 10 different versions):\n",
                "- Commit 0: `DATASET_INDEX=0` (kc1-binary)\n",
                "- Commit 1: `DATASET_INDEX=1` (usp05)\n",
                "- Commit 2: `DATASET_INDEX=2` (sleuth-ex2016)\n",
                "- Commit 3: `DATASET_INDEX=3` (calendarDOW)\n",
                "- Commit 4: `DATASET_INDEX=4` (mc2)\n",
                "- Commit 5: `DATASET_INDEX=5` (fri-c1)\n",
                "- Commit 6: `DATASET_INDEX=6` (ipums-la-99)\n",
                "- Commit 7: `DATASET_INDEX=7` (madelon)\n",
                "- Commit 8: `DATASET_INDEX=8` (mfeat-fourier)\n",
                "- Commit 9: `DATASET_INDEX=9` (robot-failures-lp5)\n",
                "\n",
                "Each run executes in a fresh process with AutoGluon 1.5.0 + NumPy<2 and finishes in ~20-35 minutes!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%bash\n",
                "# Step 1: Install AutoGluon 1.5.0 with numpy<2 into /tmp/aglibs\n",
                "python -c \"import autogluon\" 2>/dev/null || \\\n",
                "  pip install -q --target=/tmp/aglibs \"autogluon.tabular[all]==1.5.0\" \"numpy<2\" \"openml\"\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%bash\n",
                "# Step 2: Verify that child processes load numpy<2 and autogluon from /tmp/aglibs\n",
                "export PYTHONPATH=\"/tmp/aglibs:$PYTHONPATH\"\n",
                "python -c \"import numpy as np; print('NumPy version in /tmp/aglibs:', np.__version__); assert int(np.__version__.split('.')[0]) < 2; import autogluon.tabular as ag; print('AutoGluon loaded successfully! Version:', ag.__version__); import openml; print('OpenML version:', openml.__version__)\"\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%bash\n",
                "# Step 3: Setup BETA codebase and assets (ensuring latest commit)\n",
                "if [ ! -f \"experiments/run_rq3_ablations.py\" ]; then\n",
                "  rm -rf beta configs assets experiments tmp_beta\n",
                "  if ls /kaggle/input/**/beta_bundle.tar.gz 1> /dev/null 2>&1; then\n",
                "    echo \"Extracting from Kaggle dataset...\"\n",
                "    tar -xzf /kaggle/input/**/beta_bundle.tar.gz\n",
                "  elif [ -f \"beta_bundle.tar.gz\" ]; then\n",
                "    echo \"Extracting local beta_bundle.tar.gz...\"\n",
                "    tar -xzf beta_bundle.tar.gz\n",
                "  else\n",
                "    echo \"Cloning latest repository from GitHub...\"\n",
                "    git clone https://github.com/iSE-UET-VNU/BETA.git tmp_beta && cp -r tmp_beta/* . && rm -rf tmp_beta\n",
                "  fi\n",
                "fi\n",
                "python -c \"import os; assert os.path.exists('beta') and os.path.exists('experiments/run_rq3_ablations.py'); print('BETA codebase, assets, and experiments verified!')\"\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%bash\n",
                "# Step 4: Run RQ3 ablation for the chosen dataset index (0 to 9)\n",
                "export PYTHONPATH=\"/tmp/aglibs:$PYTHONPATH\"\n",
                "\n",
                "# >>> SET YOUR DATASET INDEX HERE (0 to 9) <<<\n",
                "DATASET_INDEX=0\n",
                "TIME_LIMIT=300\n",
                "\n",
                "echo \"Running ablation for dataset index $DATASET_INDEX with time_limit=${TIME_LIMIT}s...\"\n",
                "python -u experiments/run_rq3_ablations.py \\\n",
                "  --dataset-idx $DATASET_INDEX \\\n",
                "  --downstream autogluon \\\n",
                "  --time-limit $TIME_LIMIT \\\n",
                "  --output-dir .\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 5: Inspect generated results and artifacts\n",
                "import glob, json\n",
                "from pathlib import Path\n",
                "\n",
                "print(\"Artifacts in working directory:\")\n",
                "for f in sorted(glob.glob(\"*.json\") + glob.glob(\"*.csv\") + glob.glob(\"*.tex\")):\n",
                "    p = Path(f)\n",
                "    print(f\" - {p.name} ({p.stat().st_size} bytes)\")\n",
                "\n",
                "for jf in glob.glob(\"rq3_results_*.json\"):\n",
                "    print(f\"\\nContent of {jf}:\")\n",
                "    print(Path(jf).read_text())\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("experiments/rq3_ablations_kaggle.ipynb")
out_path.write_text(json.dumps(notebook, indent=2))
print(f"Generated {out_path} successfully!")
