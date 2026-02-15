"""
Wrapper to run model/Run.py from the project root directory.

Usage (from OpenCity/):
    python main.py -mode pretrain -model OpenCity -save_pretrain_path OpenCity-plus2.0.pth -batch_size 4 --embed_dim 512 --skip_dim 512 --enc_depth 6
    python main.py -mode test -model OpenCity -load_pretrain_path OpenCity-plus.pth -batch_size 2 --embed_dim 512 --skip_dim 512 --enc_depth 6

All arguments are forwarded to model/Run.py.
The working directory is automatically changed to model/ before execution.
"""
import os
import sys
import subprocess

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(project_root, 'model')
    run_script = os.path.join(model_dir, 'Run.py')

    # Change to model/ directory so all relative paths (../data, ../conf) work
    os.chdir(model_dir)

    # Forward all CLI arguments to Run.py
    result = subprocess.run(
        [sys.executable, run_script] + sys.argv[1:],
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
