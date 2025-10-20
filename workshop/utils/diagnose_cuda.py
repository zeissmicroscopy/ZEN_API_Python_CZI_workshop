import os
import sys
import subprocess
import onnxruntime as ort

def check_cuda_installation():
    """Check CUDA installation and environment."""
    print("=" * 70)
    print("CUDA Installation Diagnostics")
    print("=" * 70)
    
    # Check CUDA_PATH environment variable
    cuda_path = os.environ.get('CUDA_PATH')
    print(f"\nCUDA_PATH environment variable: {cuda_path}")
    
    # Check PATH for CUDA
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    cuda_in_path = [d for d in path_dirs if 'cuda' in d.lower()]
    print(f"CUDA directories in PATH: {cuda_in_path}")
    
    # Try to run nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("\n✅ nvidia-smi is working:")
            print(result.stdout.split('\n')[0:3])  # Show first few lines
        else:
            print(f"\n❌ nvidia-smi failed: {result.stderr}")
    except Exception as e:
        print(f"\n❌ Could not run nvidia-smi: {e}")
    
    # Try to run nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("\n✅ nvcc is working:")
            print(result.stdout.strip())
        else:
            print(f"\n❌ nvcc failed: {result.stderr}")
    except Exception as e:
        print(f"\n❌ Could not run nvcc: {e}")


def check_onnxruntime_installation():
    """Check ONNX Runtime installation details."""
    print("\n" + "=" * 70)
    print("ONNX Runtime Installation Diagnostics")
    print("=" * 70)
    
    print(f"\nONNX Runtime version: {ort.__version__}")
    print(f"ONNX Runtime location: {ort.__file__}")
    
    # Check available providers
    available_providers = ort.get_available_providers()
    print(f"\nAvailable providers: {available_providers}")
    
    # Check if CUDA provider files exist
    ort_path = os.path.dirname(ort.__file__)
    cuda_dll_path = os.path.join(ort_path, 'capi', 'onnxruntime_providers_cuda.dll')
    tensorrt_dll_path = os.path.join(ort_path, 'capi', 'onnxruntime_providers_tensorrt.dll')
    
    print(f"\nChecking ONNX Runtime provider DLLs:")
    print(f"CUDA DLL exists: {os.path.exists(cuda_dll_path)} - {cuda_dll_path}")
    print(f"TensorRT DLL exists: {os.path.exists(tensorrt_dll_path)} - {tensorrt_dll_path}")
    
    # List all files in capi directory
    capi_path = os.path.join(ort_path, 'capi')
    if os.path.exists(capi_path):
        print(f"\nFiles in ONNX Runtime capi directory:")
        for file in os.listdir(capi_path):
            if file.endswith('.dll'):
                print(f"  {file}")


def check_dependencies():
    """Check for required dependencies."""
    print("\n" + "=" * 70)
    print("Dependency Diagnostics")
    print("=" * 70)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Check for common CUDA-related DLLs in system
    import ctypes
    
    dlls_to_check = [
        'cudart64_12.dll',
        'cudart64_11.dll', 
        'curand64_10.dll',
        'cufft64_11.dll',
        'cublas64_12.dll',
        'cublas64_11.dll',
        'cudnn64_9.dll',
        'cudnn64_8.dll'
    ]
    
    print("\nChecking for CUDA/cuDNN DLLs in system:")
    for dll in dlls_to_check:
        try:
            ctypes.cdll.LoadLibrary(dll)
            print(f"✅ {dll} - Found")
        except OSError:
            print(f"❌ {dll} - Not found")


def suggest_fixes():
    """Suggest potential fixes based on the diagnostics."""
    print("\n" + "=" * 70)
    print("Suggested Fixes")
    print("=" * 70)
    
    print("\n1. Install ONNX Runtime GPU version:")
    print("   conda activate smartmic")
    print("   pip uninstall onnxruntime")
    print("   pip install onnxruntime-gpu")
    
    print("\n2. Or install via conda (recommended):")
    print("   conda activate smartmic")
    print("   conda remove onnxruntime")
    print("   conda install onnxruntime-gpu -c conda-forge")
    
    print("\n3. Check CUDA toolkit installation:")
    print("   conda activate smartmic")
    print("   conda install cudatoolkit=11.8 cudnn=8.2 -c conda-forge")
    
    print("\n4. Alternative: Install specific ONNX Runtime version:")
    print("   pip install onnxruntime-gpu==1.16.3  # Known stable version")
    
    print("\n5. Check GPU driver and CUDA compatibility:")
    print("   Run 'nvidia-smi' to check driver version")
    print("   Ensure GPU driver supports CUDA 11.8 or 12.x")


if __name__ == "__main__":
    check_cuda_installation()
    check_onnxruntime_installation() 
    check_dependencies()
    suggest_fixes()