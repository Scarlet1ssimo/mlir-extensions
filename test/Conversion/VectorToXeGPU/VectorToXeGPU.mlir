// RUN: imex-opt --split-input-file --convert-vector-to-xegpu %s -verify-diagnostics -o -| FileCheck %s

module attributes {gpu.container_module, torch.debug_module_name = "ReLU"} {
  gpu.module @forward_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @forward_kernel(%arg0: index, %arg1: memref<512x640xf32>, %arg2: f32, %arg3: vector<32xf32>, %arg4: memref<512x640xf32>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 80, 1, 1>, gpu.known_grid_size = array<i32: 512, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = arith.muli %1, %arg0 : index
      %3 = vector.transfer_read %arg1[%0, %2], %arg2 : memref<512x640xf32>, vector<32xf32>
      %4 = arith.cmpf ugt, %3, %arg3 : vector<32xf32>
      %5 = arith.select %4, %3, %arg3 : vector<32xi1>, vector<32xf32>
      vector.transfer_write %5, %arg4[%0, %2] : vector<32xf32>, memref<512x640xf32>
      gpu.return
    }
  }
}

