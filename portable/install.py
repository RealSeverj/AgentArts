"""Install the tested model dependencies in this copy of the project."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cu118"), default="cpu")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or struct.calcsize("P") != 8:
        parser.exit(2, "Use 64-bit Python 3.12 for the tested dependency set.\n")
    home = ROOT / ".venv"
    venv.EnvBuilder(with_pip=True).create(home)
    python = home / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    pip = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
    if sys.platform == "darwin":
        if args.backend != "cpu":
            parser.exit(2, "The cu118 PyTorch wheels require Windows or Linux.\n")
        torch_packages = ["torch==2.6.0", "torchvision==0.21.0"]
    else:
        torch_packages = ["--index-url", f"https://download.pytorch.org/whl/{args.backend}",
                          "torch==2.6.0+" + args.backend, "torchvision==0.21.0+" + args.backend]
    subprocess.run(pip + torch_packages, check=True)
    subprocess.run(pip + ["-r", str(ROOT / "requirements-inference.txt"),
                          "safetensors==0.8.0", "torchseg==0.0.1a4",
                          "einops==0.8.1", "opencv-python-headless==4.10.0.84"], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    print(f"Installed Python: {python}")


if __name__ == "__main__":
    main()
