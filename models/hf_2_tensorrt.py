import os
import gc
import torch
import torch_tensorrt
import argparse
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
from ament_index_python.packages import get_package_share_directory

def compile_and_save_trt_engine(
    model_dir,
    min_batch=1,
    opt_batch=128,
    max_batch=256,
    image_size=224
):
    """
    Compile and save a TensorRT engine from a HuggingFace model.
    
    Args:
        model_dir: Path to the HuggingFace model directory
        min_batch: Minimum batch size
        opt_batch: Optimal batch size
        max_batch: Maximum batch size
        image_size: Input image size (height and width)
    """
    torch.cuda.empty_cache()
    gc.collect()

    engine_path = os.path.join(model_dir, 'compiled_model.ep')
    
    print(f"Loading model from {model_dir} (make sure to have built and sourced)")
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True).to("cuda").eval()
    
    print(f"\nCompiling TensorRT engine with dynamic batch size:")
    print(f"  Min batch: {min_batch}")
    print(f"  Optimal batch: {opt_batch}")
    print(f"  Max batch: {max_batch}")
    print(f"  Image size: {image_size}x{image_size}")
    
    inputs = [
        torch_tensorrt.Input(
            min_shape=(min_batch, 3, image_size, image_size),
            opt_shape=(opt_batch, 3, image_size, image_size),
            max_shape=(max_batch, 3, image_size, image_size),
            dtype=torch.float32
        )
    ]
    
    print("\nCompiling with TensorRT (Dynamo)... (this may take several minutes)")
    trt_model = torch_tensorrt.compile(
        model,
        inputs=inputs,
        enabled_precisions={torch.float16},
        truncate_double=True,
    )
    
    print(f"\nSaving TensorRT model to: {engine_path}")
    # Save using torch.export for Dynamo models
    torch_tensorrt.save(trt_model, engine_path, inputs=inputs)
    
    print("\n✓ TensorRT engine saved successfully!")
    print(f"  Engine file: {engine_path}")
    print(f"  Dynamic batch range: [{min_batch}, {max_batch}]")
    
    return engine_path

def main():
    parser = argparse.ArgumentParser(description='Compile HuggingFace model to TensorRT engine')
    
    # Get default model directory
    pallet_processor_path = os.path.join(os.path.expanduser("~"), "ros2_ws", "src", "pallet_processor")
    default_model_dir = 'andina-dinov3-vits-triplet'
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default=default_model_dir,
        help=f'Path to the HuggingFace model directory (default: {default_model_dir})'
    )
    parser.add_argument(
        '--min-batch',
        type=int,
        default=1,
        help='Minimum batch size (default: 1)'
    )
    parser.add_argument(
        '--opt-batch',
        type=int,
        default=128,
        help='Optimal batch size (default: 128)'
    )
    parser.add_argument(
        '--max-batch',
        type=int,
        default=256,
        help='Maximum batch size (default: 256)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Input image size (default: 224)'
    )
    
    args = parser.parse_args()
    
    try:
        compile_and_save_trt_engine(
            model_dir=args.model_dir,
            min_batch=args.min_batch,
            opt_batch=args.opt_batch,
            max_batch=args.max_batch,
            image_size=args.image_size
        )
    except Exception as e:
        print(f"\n✗ Error during compilation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
