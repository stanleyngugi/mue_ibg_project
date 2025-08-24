# File: run_phase4_debug_minimal.py
# Minimal test to isolate which component is causing OOM

import torch
import gc
from mue_pipeline import MUEDiffusionPipeline
from dis_module import DISCalculator
from gloss_module import GlossCalculator

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

if __name__ == "__main__":
    print("=== Phase 4 DEBUG: Component-by-Component Testing ===")
    clear_gpu_cache()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test 1: Pipeline only
    print("\n1. Testing Pipeline Initialization...")
    print_gpu_memory()
    
    mue_pipe = MUEDiffusionPipeline(device=device, torch_dtype=torch.float16, compile_models=False)
    print("✅ Pipeline initialized")
    print_gpu_memory()
    
    # Test 2: DIS Calculator
    print("\n2. Testing DIS Calculator...")
    try:
        dis_calculator = DISCalculator(device=device, compile_models=False)
        print("✅ DIS Calculator initialized")
        print_gpu_memory()
    except Exception as e:
        print(f"❌ DIS Calculator failed: {e}")
        clear_gpu_cache()
    
    # Test 3: Gloss Calculator
    print("\n3. Testing Gloss Calculator...")
    try:
        gloss_calculator = GlossCalculator(device=device, compile_models=False)
        print("✅ Gloss Calculator initialized")
        print_gpu_memory()
    except Exception as e:
        print(f"❌ Gloss Calculator failed: {e}")
        clear_gpu_cache()
    
    # Test 4: Basic Generation (No MUE)
    print("\n4. Testing Basic Generation (No DIS/Gloss)...")
    try:
        result = mue_pipe(
            prompt="simple test image",
            height=512, width=512,  # Very small
            num_inference_steps_base=5,
            num_inference_steps_refiner=2,
            seed=42
        )
        print("✅ Basic generation succeeded")
        result['images'][0].save("debug_basic.png")
        print_gpu_memory()
    except Exception as e:
        print(f"❌ Basic generation failed: {e}")
        clear_gpu_cache()
    
    # Test 5: Gloss Only
    print("\n5. Testing Gloss Only...")
    try:
        result = mue_pipe(
            prompt="test with gloss",
            height=512, width=512,
            num_inference_steps_base=5,
            num_inference_steps_refiner=2,
            gloss_calculator=gloss_calculator,
            gloss_target="sharp focus",
            gloss_strength=0.02,
            gloss_active_start_step=3,
            seed=42
        )
        print("✅ Gloss-only generation succeeded")
        result['images'][0].save("debug_gloss_only.png")
        print_gpu_memory()
    except Exception as e:
        print(f"❌ Gloss-only generation failed: {e}")
        print(f"Error details: {e}")
        clear_gpu_cache()
    
    print("\n=== Debug Test Complete ===")
    print("Check which test failed to identify the problematic component.")
