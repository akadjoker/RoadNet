import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

 
ENGINE_PATH = 'road_model/small_model.engine'

 
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open(ENGINE_PATH, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

print("Engine carregado com sucesso.")

context = engine.create_execution_context()

# ===== Obter info de input/output =====
input_idx = engine.get_binding_index(engine[0])
output_idx = engine.get_binding_index(engine[1])

input_shape = engine.get_binding_shape(input_idx)
output_shape = engine.get_binding_shape(output_idx)

print(f"Input shape: {input_shape}")
print(f"Output shape: {output_shape}")

# ===== Preparar memória =====
input_size = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)

d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)

bindings = [int(d_input), int(d_output)]

# ===== Criar input aleatório =====
dummy_input = np.random.rand(*input_shape).astype(np.float32)

cuda.memcpy_htod(d_input, dummy_input)

 
context.execute_v2(bindings)

output = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh(output, d_output)

print("Inferência concluída com sucesso.")
print(f"Output (primeiros 10 valores): {output.flatten()[:10]}")
