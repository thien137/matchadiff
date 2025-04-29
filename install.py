import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup the environment')
    
    parser.add_argument('--env_name', type=str, default='matcha', help='Name of the conda environment to create')
    args = parser.parse_args()
    
    # Create a new conda environment
    # print(f"[INFO] Creating the conda environment {args.env_name} for MAtCha...")
    # os.system(f"conda env create -f environment.yml -n {args.env_name}")
    # print(f"[INFO] Conda environment {args.env_name} created.")
    
    # Install 2D Gaussian Splatting rasterizer
    # print(f"\n[INFO] Installing the 2D Gaussian Splatting rasterizer in the conda environment {args.env_name}...")
    # os.chdir("2d-gaussian-splatting/submodules/diff-surfel-rasterization/")
    # os.system(f"conda run -n {args.env_name} pip install -e .")
    # print(f"[INFO] 2D Gaussian Splatting rasterizer installed in the conda environment {args.env_name}.")
    
    # Install simple-knn
    # print(f"\n[INFO] Installing simple-knn in the conda environment {args.env_name}...")
    # os.chdir("../simple-knn/")
    # os.system(f"conda run -n {args.env_name} pip install -e .")
    # print(f"[INFO] simple-knn installed in the conda environment {args.env_name}.")
    
    # Install tetra-triangulation
    print(f"\n[INFO] Installing tetra-triangulation in the conda environment {args.env_name}...")
    os.chdir("./2d-gaussian-splatting/submodules/tetra-triangulation/")
    os.system(f"conda run -n {args.env_name} cmake .")
    os.system(f"conda run -n {args.env_name} export CPATH=/usr/local/cuda-11.8/targets/x86_64-linux/include:$CPATH")
    os.system(f"conda run -n {args.env_name} export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH")
    os.system(f"conda run -n {args.env_name} export PATH=/usr/local/cuda-11.8/bin:$PATH")
    os.system(f"conda run -n {args.env_name} make")
    os.system(f"conda run -n {args.env_name} pip install -e .")
    print(f"[INFO] tetra-triangulation installed in the conda environment {args.env_name}.")
    os.chdir("../../../")
        
    # Install ASMK
    # print(f"\n[INFO] Installing ASMK...")
    # os.chdir("./mast3r/asmk/cython")
    # os.system(f"conda run -n {args.env_name} cythonize *.pyx")
    # os.chdir("../")
    # os.system(f"conda run -n {args.env_name} pip install .")
    # print("[INFO] ASMK installed.")
    
    # Compile cuda kernels for RoPE
    # print(f"\n[INFO] Compiling cuda kernels for RoPE...")
    # os.chdir("../dust3r/croco/models/curope/")
    # os.system(f"conda run -n {args.env_name} python setup.py build_ext --inplace")
    # print("[INFO] RoPE cuda kernels compiled.")
    
    # os.chdir("../../../../../")
    # print("[INFO] MAtCha installation complete.")