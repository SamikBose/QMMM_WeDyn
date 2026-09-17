"""Run standalone gates with a machine-readable report; preserve failed reports."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--nve',action='store_true',help='Include the 1 ps timestep-convergence test')
args = parser.parse_args()
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
Path('validation').mkdir(exist_ok=True)
command = [sys.executable,'-m','pytest','-q','-s',f'--junitxml=validation/standalone-{stamp}.xml']
if not args.nve:
    command += ['-m','not dynamics']
sys.exit(subprocess.call(command))
