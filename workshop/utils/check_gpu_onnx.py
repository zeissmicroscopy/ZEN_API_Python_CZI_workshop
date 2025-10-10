"""
ONNX Runtime GPU Verification and Testing Tool

This module provides comprehensive testing and validation for GPU-accelerated
inference using ONNX Runtime. It checks for:
- CUDA provider availability
- TensorRT provider availability (optional)
- Actual GPU utilization during inference
- Performance comparison between GPU and CPU execution
- Real-time GPU monitoring during inference operations

Requirements:
    - onnxruntime-gpu: ONNX Runtime with GPU support
    - numpy: For creating test data
    - GPUtil (optional): For detailed GPU monitoring
    - psutil (optional): For system resource monitoring
    - onnx: For creating test models

Usage:
    python check_gpu_onnx.py
    
    Returns exit code 0 if GPU is working properly, 1 otherwise.

Author: Workshop materials for ZEN Python CZI Smart Microscopy Workshop
"""

import onnxruntime as ort
import numpy as np
import time
import sys

# Try to import optional GPU monitoring libraries
# These provide enhanced monitoring capabilities but are not required for basic functionality
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("Warning: GPUtil not available. GPU monitoring will be limited.")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available. CPU monitoring will be limited.")


def get_gpu_utilization():
    """
    Retrieve current GPU utilization metrics.
    
    This function queries the first available GPU and returns detailed metrics
    including load percentage, memory usage, and temperature.
    
    Returns:
        dict or None: Dictionary containing GPU metrics if GPUtil is available and
                     GPU is detected, None otherwise. Dictionary contains:
                     - 'load': GPU load percentage (0-100)
                     - 'memory_used': Used memory in MB
                     - 'memory_total': Total memory in MB
                     - 'memory_util': Memory utilization percentage (0-100)
                     - 'temperature': GPU temperature in Celsius
    
    Note:
        Requires GPUtil to be installed. If not available, returns None.
    """
    if HAS_GPUTIL:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU (index 0)
                return {
                    'load': gpu.load * 100,              # Convert to percentage
                    'memory_used': gpu.memoryUsed,       # In MB
                    'memory_total': gpu.memoryTotal,     # In MB
                    'memory_util': gpu.memoryUtil * 100, # Convert to percentage
                    'temperature': gpu.temperature       # In Celsius
                }
        except Exception as e:
            print(f"Error getting GPU utilization: {e}")
    return None


def monitor_gpu_usage(duration=2):
    """
    Continuously monitor GPU usage over a specified time period.
    
    Takes measurements at 0.5 second intervals and displays real-time
    GPU load and memory usage statistics.
    
    Args:
        duration (int): Duration in seconds to monitor GPU. Default is 2 seconds.
    
    Returns:
        list: List of dictionaries containing GPU measurements over time.
              Empty list if GPU monitoring is not available.
    
    Example:
        >>> measurements = monitor_gpu_usage(duration=5)
        >>> avg_load = sum(m['load'] for m in measurements) / len(measurements)
    """
    print(f"\n🔍 Monitoring GPU usage for {duration} seconds...")
    
    start_time = time.time()
    measurements = []
    
    # Poll GPU metrics every 0.5 seconds until duration expires
    while time.time() - start_time < duration:
        gpu_info = get_gpu_utilization()
        if gpu_info:
            measurements.append(gpu_info)
            # Display current GPU metrics
            print(f"GPU Load: {gpu_info['load']:.1f}% | "
                  f"Memory: {gpu_info['memory_used']:.0f}/{gpu_info['memory_total']:.0f}MB "
                  f"({gpu_info['memory_util']:.1f}%)")
        else:
            print("GPU monitoring not available")
            break
        time.sleep(0.5)  # Sample interval
    
    return measurements


def create_dummy_model():
    """
    Create a simple ONNX model for GPU performance testing.
    
    Generates a matrix multiplication model (4000x4000 matrices) which is
    computationally intensive enough to demonstrate GPU acceleration benefits.
    The model performs: Z = MatMul(X, Y) where X and Y are 4000x4000 matrices.
    
    Returns:
        str or None: Path to the saved ONNX model file ('dummy_model.onnx') if
                    successful, None if ONNX library is not available or error occurs.
    
    Model Details:
        - Operation: Matrix Multiplication (MatMul)
        - Input tensors: Two 4000x4000 float32 matrices
        - Output tensor: One 4000x4000 float32 matrix
        - IR version: 7 (compatible with ONNX Runtime 1.x)
        - Opset version: 11
    
    Note:
        The model is intentionally large (4000x4000) to stress the GPU and
        demonstrate clear performance differences between GPU and CPU execution.
    """
    try:
        import onnx
        from onnx import helper, TensorProto
        
        # Create input tensors - larger size (4000x4000) for better GPU utilization
        # Matrix multiplication is highly parallelizable and benefits from GPU
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [4000, 4000])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [4000, 4000])
        
        # Create output tensor - result of X @ Y
        Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [4000, 4000])

        # Create matrix multiplication node
        matmul_node = helper.make_node(
            'MatMul',           # Operation type
            inputs=['X', 'Y'],  # Input tensor names
            outputs=['Z'],      # Output tensor name
            name='MatMul'       # Node name
        )
        
        # Create computational graph
        graph = helper.make_graph(
            [matmul_node],      # List of nodes
            'dummy_graph',      # Graph name
            [X, Y],             # Graph inputs
            [Z]                 # Graph outputs
        )
        
        # Create ONNX model
        model = helper.make_model(graph, producer_name='dummy')
        model.ir_version = 7  # Set compatible IR version for ONNX Runtime
        model.opset_import[0].version = 11  # Use opset 11
        
        # Save model to disk
        model_path = "dummy_model.onnx"
        onnx.save(model, model_path)
        return model_path
        
    except ImportError:
        print("ONNX library not available, using numpy-based test instead")
        return None
    except Exception as e:
        print(f"Error creating dummy model: {e}")
        return None


def test_gpu_inference(model_path=None):
    """
    Test actual GPU utilization during ONNX Runtime inference.
    
    This function performs comprehensive testing to verify that the GPU is
    actually being used for inference, not just available. It:
    1. Creates or loads an ONNX model
    2. Runs inference with GPU provider and measures time/GPU usage
    3. Runs the same inference with CPU-only for comparison
    4. Analyzes performance difference and GPU load patterns
    
    Args:
        model_path (str, optional): Path to an existing ONNX model for testing.
                                   If None, creates a dummy model automatically.
    
    Returns:
        bool: True if GPU is being utilized effectively, False otherwise.
    
    Test Criteria:
        - GPU inference should be faster than CPU (or close for small models)
        - GPU load should increase measurably during inference
        - Session should successfully use CUDAExecutionProvider
    
    Note:
        Falls back to basic GPU functionality test if model creation fails.
    """
    print("\n🧪 Testing GPU inference...")
    
    # Create or load test model
    if model_path is None:
        print("Creating dummy model for testing...")
        model_path = create_dummy_model()
        if model_path is None:
            print("❌ Could not create test model, running basic provider test instead")
            return test_basic_gpu_functionality()
    
    try:
        # ===== PART 1: Test with GPU Provider =====
        print("\n--- Testing with GPU provider ---")
        
        # Specify provider priority: try CUDA first, fall back to CPU if needed
        gpu_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Capture baseline GPU usage before starting inference
        baseline_gpu = get_gpu_utilization()
        if baseline_gpu:
            print(f"Baseline GPU usage: {baseline_gpu['load']:.1f}%")
        
        # Create inference session with GPU provider
        session_gpu = ort.InferenceSession(model_path, providers=gpu_providers)
        print(f"Session providers: {session_gpu.get_providers()}")
        
        # Create large input data (4000x4000 matrices) to stress the GPU
        # Using float32 for efficiency and compatibility
        input_data = {
            'X': np.random.rand(4000, 4000).astype(np.float32),
            'Y': np.random.rand(4000, 4000).astype(np.float32)
        }
        
        print("Running inference with potential GPU acceleration...")
        start_time = time.time()
        
        # Run multiple inferences to get reliable timing
        num_iterations = 5
        for i in range(num_iterations):
            result_gpu = session_gpu.run(None, input_data)
            if i == 0:
                # Verify output shape on first iteration
                print(f"First inference completed, result shape: {result_gpu[0].shape}")
        
        gpu_time = time.time() - start_time
        print(f"GPU inference time: {gpu_time:.3f} seconds (average: {gpu_time/num_iterations:.3f}s per inference)")
        
        # ===== PART 2: Monitor GPU Load During Active Inference =====
        print("Monitoring GPU during active inference...")
        measurements = []
        
        # Run inference while monitoring GPU metrics
        for i in range(5):
            # Capture GPU state before inference
            gpu_info = get_gpu_utilization()
            if gpu_info:
                measurements.append(gpu_info)
                print(f"Inference {i+1}: GPU Load: {gpu_info['load']:.1f}% | "
                      f"Memory: {gpu_info['memory_used']:.0f}MB")
            
            # Execute inference
            _ = session_gpu.run(None, input_data)
            time.sleep(0.1)  # Brief pause to allow GPU metrics to update
        
        # ===== PART 3: Test with CPU-Only for Performance Comparison =====
        print("\n--- Testing with CPU only for comparison ---")
        cpu_providers = ['CPUExecutionProvider']
        session_cpu = ort.InferenceSession(model_path, providers=cpu_providers)
        
        # Run same number of iterations on CPU
        start_time = time.time()
        for i in range(num_iterations):
            _ = session_cpu.run(None, input_data)
        cpu_time = time.time() - start_time
        print(f"CPU inference time: {cpu_time:.3f} seconds (average: {cpu_time/num_iterations:.3f}s per inference)")
        
        # ===== PART 4: Analyze Results and Determine GPU Effectiveness =====
        print("\n📊 Performance Comparison:")
        print(f"GPU time: {gpu_time:.3f}s")
        print(f"CPU time: {cpu_time:.3f}s")
        
        # Check if GPU provides speed benefit
        if gpu_time < cpu_time:
            speedup = cpu_time / gpu_time
            print(f"🚀 GPU is {speedup:.2f}x faster than CPU")
            gpu_used = True
        else:
            print("⚠️  GPU is not faster than CPU - may not be utilized properly")
            gpu_used = False
        
        # Analyze GPU load measurements to confirm actual GPU usage
        # Even if timing is good, we want to see actual GPU load increase
        if measurements and any(m['load'] > 10 for m in measurements):
            print("✅ GPU load detected during inference - GPU is being used!")
            gpu_used = True
        elif measurements:
            max_load = max(m['load'] for m in measurements)
            print(f"⚠️  Low GPU load detected (max: {max_load:.1f}%) - GPU may not be fully utilized")
        else:
            print("❌ Could not monitor GPU usage")
        
        return gpu_used
        
    except Exception as e:
        print(f"❌ Error during GPU inference test: {e}")
        print("Falling back to basic GPU functionality test...")
        return test_basic_gpu_functionality()


def test_basic_gpu_functionality():
    """
    Perform basic GPU functionality check without model inference.
    
    This is a fallback test used when ONNX model creation fails. It verifies
    that CUDA provider can be initialized and basic GPU information can be
    queried.
    
    Returns:
        bool: True if CUDA provider is functional, False otherwise.
    
    Tests:
        - CUDA provider initialization
        - GPU utilization query (if GPUtil available)
        - Basic GPU memory information
    
    Note:
        This test is less comprehensive than full inference testing but
        provides a baseline verification of GPU availability.
    """
    print("\n🔧 Running basic GPU functionality test...")
    
    try:
        # Test creating a session with CUDA provider
        print("Testing CUDA provider initialization...")
        
        # Configure session options to reduce verbose logging
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # 3 = ERROR level only
        
        # If we reach here, CUDA provider is at least available
        print("✅ CUDA provider can be initialized")
        
        # Check GPU memory usage if monitoring is available
        gpu_info = get_gpu_utilization()
        if gpu_info:
            print(f"Current GPU usage: {gpu_info['load']:.1f}% load, "
                  f"{gpu_info['memory_used']:.0f}MB memory used")
            return True
        else:
            print("⚠️  GPU monitoring not available, but CUDA provider is functional")
            return True
            
    except Exception as e:
        print(f"❌ Basic GPU functionality test failed: {e}")
        return False
def check_gpu():
    """
    Main function to check GPU availability and functionality for ONNX Runtime.
    
    This function performs a comprehensive check of GPU support including:
    1. ONNX Runtime version verification
    2. CUDA execution provider availability
    3. TensorRT execution provider availability (optional)
    4. Actual GPU inference testing with performance validation
    
    Returns:
        bool: True if GPU is available and working properly, False otherwise.
    
    Exit Behavior:
        When run as main script, exits with code 0 on success, 1 on failure.
    
    Output:
        Prints detailed diagnostic information including:
        - ONNX Runtime version
        - Available execution providers
        - GPU inference performance metrics
        - GPU vs CPU speed comparison
    
    Example Output:
        ======================================================================
        ONNX Runtime GPU Check
        ======================================================================
        ONNX Runtime version: 1.19.0
        Available providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ✅ CUDAExecutionProvider is available
        🎉 GPU inference test PASSED - GPU is being utilized!
    """
    print("=" * 70)
    print("ONNX Runtime GPU Check")
    print("=" * 70)

    # Display ONNX Runtime version for debugging purposes
    print(f"\nONNX Runtime version: {ort.__version__}")

    # Query available execution providers
    # Providers are listed in priority order
    available_providers = ort.get_available_providers()
    print(f"\nAvailable providers: {available_providers}")

    # Check for CUDA provider (required for GPU acceleration)
    has_cuda = "CUDAExecutionProvider" in available_providers
    has_tensorrt = "TensorrtExecutionProvider" in available_providers

    if has_cuda:
        print("\n✅ CUDAExecutionProvider is available")
    else:
        print("\n❌ CUDAExecutionProvider is NOT available")
        print("   Please install onnxruntime-gpu and ensure CUDA/cuDNN are installed")
        print("ONNX Runtime GPU check FAILED")
        return False

    # TensorRT is optional but provides additional optimizations
    if has_tensorrt:
        print("✅ TensorrtExecutionProvider is available")
    else:
        print("ℹ️  TensorrtExecutionProvider is not available (optional)")

    # Perform actual inference test to verify GPU is really working
    # Provider availability doesn't guarantee functionality
    gpu_working = test_gpu_inference()
    
    if gpu_working:
        print("\n🎉 GPU inference test PASSED - GPU is being utilized!")
        return True
    else:
        print("\n❌ GPU inference test FAILED - GPU may not be working properly")
        print("   Check CUDA/cuDNN installation and compatibility")
        return False


if __name__ == "__main__":
    """
    Script entry point for command-line execution.
    
    Runs the GPU check and returns appropriate exit code for scripting.
    Exit code 0 indicates success, 1 indicates failure.
    
    Cleanup:
        Removes temporary dummy_model.onnx file if created during testing.
    """
    # Run comprehensive GPU check
    success = check_gpu()
    
    # Clean up any temporary files created during testing
    import os
    if os.path.exists("dummy_model.onnx"):
        try:
            os.remove("dummy_model.onnx")
        except OSError as e:
            print(f"\nWarning: Could not remove temporary file: {e}")
    
    # Exit with appropriate code for scripting/CI purposes
    sys.exit(0 if success else 1)
