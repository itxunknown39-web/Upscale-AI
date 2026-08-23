import os
import sys
import argparse
import glob
import cv2
import torch
import numpy as np
from PIL import Image

# Ensure torchvision functional_tensor backward compatibility for basicsr
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except Exception:
    pass

# Ensure basicsr module shim & RRDBNet are available
from scripts.rrdbnet import RRDBNet, register_basicsr_shim
register_basicsr_shim()

from scripts.model_manager import ensure_model_weights, find_model_weights
from scripts.upscaler import tile_process

def main():
    parser = argparse.ArgumentParser(description="Real-ESRGAN Inference")
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-n', '--model_name', type=str, default='RealESRGAN_x4plus', help='Model name')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=2.0, help='Upsampling scale (2.0 = Stock Ready default, 3.0 = High, 4.0 = Maximum)')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=400, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=10, help='Pre padding size at each border')
    parser.add_argument('--half', action='store_true', help='Use half precision (FP16) during inference')
    parser.add_argument('--ext', type=str, default='auto', help='Image extension. Options: auto | jpg | png')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the pre-trained model')
    parser.add_argument('--quality', type=int, default=95, help='JPEG output quality (1-100)')
    args = parser.parse_args()

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_half = args.half and (device.type == 'cuda')

    # Define model architecture
    if args.model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    # Locate model weights cleanly
    model_path = args.model_path
    if not model_path or not os.path.exists(model_path):
        success, found_path, err_msg = ensure_model_weights(args.model_name, auto_download=True)
        if success and found_path:
            model_path = found_path
        else:
            print(f"[Error] Model weight file for '{args.model_name}' not found: {err_msg}")
            sys.exit(1)


    print(f"Loading weights from: {model_path} (Device: {device}, FP16: {use_half})")
    loadnet = torch.load(model_path, map_location=torch.device('cpu'))
    if 'params_ema' in loadnet:
        state_dict = loadnet['params_ema']
    elif 'params' in loadnet:
        state_dict = loadnet['params']
    else:
        state_dict = loadnet

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)
    if use_half:
        model = model.half()

    os.makedirs(args.output, exist_ok=True)

    # Discover inputs
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        if extension.lower() not in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff']:
            continue

        print(f"Processing ({idx+1}/{len(paths)}): {path}")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Failed to read image {path}")
            continue

        has_alpha = len(img.shape) == 3 and img.shape[2] == 4
        if has_alpha:
            alpha = img[:, :, 3]
            img_rgb = img[:, :, :3]
        else:
            alpha = None
            img_rgb = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img_float = img_rgb.astype(np.float32) / 255.0
        img_t = torch.from_numpy(np.transpose(img_float[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_t = img_t.unsqueeze(0).to(device)
        if use_half:
            img_t = img_t.half()

        tile_size = args.tile if args.tile > 0 else 400
        output_t = tile_process(model, img_t, tile_size=tile_size, tile_pad=args.tile_pad, scale=4)

        output_np = output_t.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output_np[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        if args.outscale != 4.0 and args.outscale > 0:
            h, w = img_rgb.shape[:2]
            target_w = int(w * args.outscale)
            target_h = int(h * args.outscale)
            output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # Output format & path
        if args.ext == 'auto':
            ext = extension[1:]
        else:
            ext = args.ext

        suffix = f"_{args.suffix}" if args.suffix else ""
        save_path = os.path.join(args.output, f"{imgname}{suffix}.{ext}")

        if has_alpha and ext.lower() == "png":
            h_out, w_out = output.shape[:2]
            alpha_resized = cv2.resize(alpha, (w_out, h_out), interpolation=cv2.INTER_LANCZOS4)
            output = cv2.merge([output[:, :, 0], output[:, :, 1], output[:, :, 2], alpha_resized])

        if ext.lower() in ['jpg', 'jpeg']:
            if len(output.shape) == 3 and output.shape[2] == 4:
                output = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(save_path, output, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        else:
            cv2.imwrite(save_path, output)

        print(f"✓ Saved upscaled image: {save_path}")

    print("Inference completed successfully!")

if __name__ == '__main__':
    main()
