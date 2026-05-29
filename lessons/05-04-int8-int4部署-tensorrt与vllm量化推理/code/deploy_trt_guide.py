#!/usr/bin/env python3
"""
TensorRT Setup Guide and Inference Example

====================================================================
                    TENSORRT DEPLOYMENT GUIDE
====================================================================

TensorRT INT8 deployment requires complex CUDA setup that cannot be
fully automated. This guide provides step-by-step instructions for
setting up TensorRT and building INT8 engines.

====================================================================
SECTION 1: TENSORRT INSTALLATION
====================================================================

Prerequisites:
-------------
1. NVIDIA GPU with CUDA Compute Capability 7.0+ (Volta or newer)
2. Ubuntu 20.04/22.04 or CentOS 8 (other OSes see NVIDIA docs)
3. CUDA Toolkit 11.8+ installed
4. cuDNN 8.0+ installed
5. Python 3.8+

Installation Steps:
------------------
1. Visit: https://developer.nvidia.com/tensorrt/download
2. Download the TensorRT tar file for your CUDA version
3. Extract the tar file:
   tar xzf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz

4. Add to PATH and LD_LIBRARY_PATH:
   export PATH=$PATH:/path/to/TensorRT/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/TensorRT/lib

5. Install Python bindings:
   pip install tensorrt

6. Verify installation:
   python -c "import tensorrt as trt; print(f'TensorRT version: {trt.__version__}')"

====================================================================
SECTION 2: ONNX EXPORT FROM PYTORCH
====================================================================

Before using TensorRT, you must export your PyTorch model to ONNX format.

export_onnx.py:
---------------
#!/usr/bin/env python3
import torch
import torch.onnx

def export_model_to_onnx(
    model,
    model_path: str,
    output_path: str,
    input_shape: tuple,
    opset_version: int = 17,
):
    \"\"\"
    Export a PyTorch model to ONNX format.

    Args:
        model: The PyTorch model
        model_path: Path to save the ONNX model
        input_shape: Shape of input tensor (excluding batch)
        opset_version: ONNX opset version
    \"\"\"
    model.eval()

    # Create dummy input matching your model's expected input
    # For LLaMA-style models, typically: (batch, seq_len)
    dummy_input = torch.randn(*input_shape, device=next(model.parameters()).device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        },
        verbose=False,
    )
    print(f"Model exported to {output_path}")

Example usage:
-------------
from export_onnx import export_model_to_onnx
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-model")
export_model_to_onnx(
    model=model,
    model_path="your-model",
    output_path="model.onnx",
    input_shape=(1, 512),
)

====================================================================
SECTION 3: BUILDING TENSORRT INT8 ENGINE
====================================================================

The INT8 engine building process requires:
1. Parsing the ONNX model
2. Creating an INT8 calibration data loader
3. Configuring the builder for INT8 mode
4. Building and serializing the engine
"""

CODE_GUIDE = '''
build_trt_engine.py:
--------------------
#!/usr/bin/env python3
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTINT8EngineBuilder:
    """
    Builds TensorRT INT8 engines from ONNX models.
    
    Workflow:
    1. Parse ONNX model into TRT network
    2. Create calibrator with representative dataset
    3. Configure builder for INT8 mode
    4. Build and serialize engine
    """
    
    def __init__(self, onnx_model_path: str, logger_level=trt.Logger.WARNING):
        self.onnx_model_path = onnx_model_path
        self.logger = trt.Logger(logger_level)
        self.builder = trt.Builder(self.logger)
        self.network = None
        self.config = None
        self.engine = None
    
    def parse_onnx(self, explicit_batch=True):
        \"\"\"Parse ONNX model into TRT network.\"\"\"
        network_flags = 0
        if explicit_batch:
            network_flags = 1 << (trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        
        self.network = self.builder.create_network(network_flags)
        parser = trt.OnnxParser(self.network, self.logger)
        
        with open(self.onnx_model_path, "rb") as f:
            success = parser.parse(f.read())
        
        if not success:
            raise RuntimeError("Failed to parse ONNX model")
        
        print(f"ONNX model parsed successfully. Layers: {self.network.num_layers}")
    
    def set_calibrator(self, calibration_data: list):
        \"\"\"
        Set up INT8 calibrator with calibration data.
        
        Args:
            calibration_data: List of numpy arrays, each being a calibration input
        \"\"\"
        self.config = self.builder.create_builder_config()
        
        # Create calibrator
        # TensorRT supports different calibration algorithms:
        # - IInt8Calibrator.MINMAX: Simple min-max scaling
        # - IInt8Calibrator.ENTROPY: KL divergence (default, most accurate)
        # - IInt8Calibrator.LEGACY: Legacy calibration
        
        calibrator = EntropyCalibrator(calibration_data)
        
        self.config.int8_mode = True
        self.config.int8_calibrator = calibrator
        
        # Set layer precision (optional, TRT infers from calibrator)
        self.config.set_quantization_config(
            trt.QuantizationConfig(
                quant_proj=trt.QuantizationProjecting.PROJECT_WEIGHTS,
                op_precision=trt.TensorRTPrecision.INT8,
            )
        )
        
        print("INT8 calibration configured")
    
    def build_engine(self, output_path: str):
        \"\"\"Build and serialize the TensorRT engine.\"\"\"
        if self.network is None:
            raise RuntimeError("Network not parsed. Call parse_onnx first.")
        
        if self.config is None:
            # Create default config if no calibrator set
            self.config = self.builder.create_builder_config()
            self.config.int8_mode = True
        
        # Build serialized engine
        serialized_engine = self.builder.build_serialized_network(self.network, self.config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save to file
        with open(output_path, "wb") as f:
            f.write(serialized_engine)
        
        print(f"Engine saved to {output_path}")
        
        # Deserialize for later use
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        return self.engine


class EntropyCalibrator(trt.IInt8Calibrator):
    """
    INT8 calibrator using KL divergence minimization.
    
    This calibrator:
    1. Runs inference with FP32 on calibration data
    2. Collects activation distributions
    3. Finds optimal scale factors to minimize KL divergence
       between FP32 and quantized distributions
    \"\"\"
    
    def __init__(self, calibration_data: list, batch_size=1):
        trt.IInt8Calibrator.__init__(self)
        
        self.calibration_data = calibration_data
        self.batch_size = batch_size
        self.current_index = 0
        
        # Allocate GPU buffers
        self.d_input = None
        self.d_output = None
        self.h_input = None
        self.h_output = None
        
        # For simplicity, using first sample shape
        # In production, should handle variable shapes
        if len(calibration_data) > 0:
            self.input_shape = calibration_data[0].shape
            self.dtype = calibration_data[0].dtype
        
        self._allocate_buffers()
    
    def _allocate_buffers(self):
        \"\"\"Allocate CUDA buffers for calibration.\"\"\"
        if len(self.calibration_data) == 0:
            return
        
        n_bytes = self.calibration_data[0].nbytes
        self.h_input = cuda.pagelocked_empty(n_bytes // 4, dtype=self.dtype)
        self.h_output = cuda.pagelocked_empty(n_bytes // 4, dtype=np.float32)
        
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
    
    def get_batch(self, names):
        \"\"\"Get next batch of calibration data.\"\"\"
        if self.current_index >= len(self.calibration_data):
            return None
        
        batch = self.calibration_data[self.current_index]
        self.current_index += 1
        
        # Copy to pagelocked buffer
        np.copyto(self.h_input, batch.ravel())
        cuda.memcpy_htod(self.d_input, self.h_input)
        
        return [int(self.d_input)]
    
    def get_batch_size(self):
        \"\"\"Return batch size for calibration.\"\"\"
        return self.batch_size
    
    def get_quantization_param(self, tensor_name):
        \"\"\"
        Return quantization parameters for a tensor.
        
        Returns (scale, min, max) for the tensor.
        In production, this computes actual KL-optimized scales.
        \"\"\"
        # Simplified: would compute actual scales from calibration data
        return np.float32(0.01), np.float32(-128.0), np.float32(127.0)
    
    def read_calibration_cache(self):
        \"\"\"Read cached calibration parameters if available.\"\"\"
        # Could cache calibration results for faster rebuilds
        return None
    
    def write_calibration_cache(self, cache):
        \"\"\"Write calibration cache for future use.\"\"\"
        pass


====================================================================
SECTION 4: TENSORRT INT8 INFERENCE
====================================================================

Once you have a built engine, use it for inference:

inference_trt.py:
----------------
#!/usr/bin/env python3
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTInference:
    """
    TensorRT INT8 inference engine.
    
    Usage:
        engine = TensorRTInference("model.engine")
        output = engine.infer(input_ids)
    \"\"\"
    
    def __init__(self, engine_path: str):
        \"\"\"Load and initialize TensorRT engine.\"\"\"
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate GPU buffers
        self.d_input = None
        self.d_output = None
        
        # Get I/O bindings
        self.input_binding_idx = self.engine.get_binding_index("input_ids")
        self.output_binding_idx = self.engine.get_binding_index("logits")
        
        # Allocate buffers based on engine dimensions
        self._allocate_buffers()
    
    def _allocate_buffers(self):
        \"\"\"Allocate GPU memory for input/output tensors.\"\"\"
        # For production, should query actual shapes from engine
        # These are placeholder sizes
        input_size = 1 * 512 * 4  # batch * seq_len * 4 bytes (int32)
        output_size = 1 * 512 * 32000 * 4  # batch * seq_len * vocab * 4 bytes
        
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)
        
        self.bindings = [int(self.d_input), int(self.d_output)]
    
    def infer(self, input_ids: np.ndarray, seq_len: int = 512) -> np.ndarray:
        \"\"\"
        Run inference with TensorRT engine.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            seq_len: Sequence length for this inference
            
        Returns:
            Logits array (batch, seq_len, vocab_size)
        \"\"\"
        # Ensure input is contiguous
        input_ids = np.ascontiguousarray(input_ids.astype(np.int32))
        
        # Create CUDA stream
        stream = cuda.Stream()
        
        # Copy input to GPU
        cuda.memcpy_htod_async(self.d_input, input_ids, stream)
        
        # Execute inference
        self.context.execute_async_v3(
            bindings=self.bindings,
            stream_handle=stream.handle,
        )
        
        # Copy output back to CPU
        output_shape = (1, seq_len, 32000)  # Adjust vocab_size as needed
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, stream)
        
        # Wait for completion
        stream.synchronize()
        
        return output
    
    def infer_batch(self, input_ids_batch: list, seq_lens: list) -> list:
        \"\"\"Run batch inference.\"\"\"
        outputs = []
        for i in range(len(input_ids_batch)):
            output = self.infer(input_ids_batch[i], seq_lens[i])
            outputs.append(output)
        return outputs


====================================================================
SECTION 5: PERFORMANCE TIPS AND TROUBLESHOOTING
====================================================================

Performance Tips:
-----------------
1. Use FP16 instead of INT8 if accuracy is critical
2. Ensure calibration data is representative of production data
3. Use larger batch sizes for better GPU utilization
4. Enable CUDA graph for non-batch inference: config.cuda_graph_enable()
5. Consider using TensorRT plugins for custom layers

Troubleshooting:
---------------
1. "TensorRT engine is null": Check if engine built successfully
2. "Binding index out of range": Verify input/output names match ONNX
3. "Calibration failed": Check calibration data shape matches ONNX input
4. "CUDA out of memory": Reduce batch size or max_seq_len
5. "ONNX parser error": Validate ONNX model with trtexec or Netron

Build Time Tips:
---------------
- Calibration is the slowest step (can take 10+ minutes)
- Cache calibration results for faster rebuilds
- Use INT8 fallback to FP16 for non-critical layers

====================================================================
SECTION 6: COMPLETE EXAMPLE WORKFLOW
====================================================================

Full workflow example:

    # Step 1: Export PyTorch to ONNX
    from export_onnx import export_model_to_onnx
    export_model_to_onnx(model, "model", "model.onnx", (1, 512))
    
    # Step 2: Prepare calibration data
    import numpy as np
    calibration_data = [np.random.randint(0, 32000, (1, 512)).astype(np.int32) 
                        for _ in range(300)]
    
    # Step 3: Build TensorRT engine
    from build_trt_engine import TensorRTINT8EngineBuilder
    builder = TensorRTINT8EngineBuilder("model.onnx")
    builder.parse_onnx()
    builder.set_calibrator(calibration_data)
    engine = builder.build_engine("model_int8.engine")
    
    # Step 4: Run inference
    from inference_trt import TensorRTInference
    trt_engine = TensorRTInference("model_int8.engine")
    input_ids = np.random.randint(0, 32000, (1, 512)).astype(np.int32)
    output = trt_engine.infer(input_ids)
    
    print(f"Output shape: {output.shape}")

====================================================================
ALTERNATIVE: USING TENSORRT AS A SERVICE (TRTEXE)
====================================================================

For production deployments, NVIDIA recommends using TRTIS (TensorRT
Inference Server) or trtexec for benchmarking. Example:

    # Benchmark with trtexec
    trtexec --onnx=model.onnx \
            --int8 \
            --calib=./calibration_data \
            --engine=model_int8.engine \
            --saveEngine=model_int8.engine
    
    # Or run inference
    trtexec --engine=model_int8.engine \
           --input=input_ids \
           --output=logits

For production serving, consider:
- NVIDIA Triton Inference Server
- TensorFlow Serving with TensorRT backend
- ONNX Runtime with TensorRT execution provider

====================================================================
'''

# Example code snippets for direct use:

EXAMPLE_EXPORT = """
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model.eval()

# Dummy input for tracing
dummy_input = torch.randint(0, 32000, (1, 512))

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "llama2_7b.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input_ids"],
    output_names=["logits"],
)
"""

EXAMPLE_BUILD = """
import tensorrt as trt
import numpy as np

# Calibration data (representative samples)
calibration_data = [np.random.randint(0, 32000, (1, 512)).astype(np.int32) 
                    for _ in range(300)]

# Build engine
builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
network = builder.create_network(1 << trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))

with open("llama2_7b.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.int8_mode = True
config.int8_calibrator = calibrator  # Your calibrator here

engine = builder.build_serialized_network(network, config)
"""

if __name__ == "__main__":
    print(__doc__)
