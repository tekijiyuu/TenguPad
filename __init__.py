from __future__ import annotations

import torch
import comfy.utils
from comfy_api.latest import IO, ComfyExtension
from typing_extensions import override
import cv2
import numpy as np


class TenguPad(IO.ComfyNode):
    """
    Advanced image resizing with flexible padding/cropping options, RGB color control,
    and content scaling within padded space.
    Preserves aspect ratio with customizable padding position, true RGB padding color (0-255),
    multiple aspect ratio modes, and adjustable content scale inside padding.
    """
    
    # Preset colors in RGB (0-255) format
    PRESETS = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
    }
    
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TenguPad",
            display_name="TenguPad v0.2",
            search_aliases=["fit image", "letterbox", "rgb pad", "aspect ratio resize", "content scale"],
            category="image/transform",
            description=(
                "Resize image while preserving aspect ratio, then pad/crop to target dimensions.\n"
                "Features true RGB padding color control (0-255), customizable position, aspect modes,\n"
                "and CONTENT SCALE to adjust image size within the padded canvas."
            ),
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("target_width", default=512, min=1, max=8192, step=1),
                IO.Int.Input("target_height", default=512, min=1, max=8192, step=1),
                IO.Combo.Input(
                    "aspect_ratio_mode",
                    options=["pad", "crop", "stretch"],
                    default="pad",
                    tooltip=(
                        "• pad: Preserve aspect ratio with padding\n"
                        "• crop: Preserve aspect ratio by cropping overflow\n"
                        "• stretch: Ignore aspect ratio, stretch to fit"
                    )
                ),
                IO.Float.Input(
                    "content_scale",
                    default=1.0,
                    min=0.01,
                    max=2.0,
                    step=0.01,
                    display_name="Content Scale",
                    tooltip=(
                        "Scale image content within padded space:\n"
                        "• < 1.0: Make image smaller inside padding (adds extra padding)\n"
                        "• = 1.0: Normal padding behavior\n"
                        "• > 1.0: Enlarge content (may cause cropping in pad mode)"
                    )
                ),
                IO.Combo.Input(
                    "padding_position",
                    options=["center", "top-left", "top-right", "bottom-left", "bottom-right", "center-top", "center-bottom"],
                    default="center",
                    tooltip="Position of content within padded canvas"
                ),
                IO.Combo.Input(
                    "interpolation",
                    options=["lanczos", "bicubic", "bilinear", "area", "nearest-exact"],
                    default="lanczos"
                ),
                # Feathering option for mask
                IO.Int.Input(
                    "feather_pixels", 
                    default=0, 
                    min=0, 
                    max=100, 
                    step=1,
                    display_name="Feather Mask Edges (pixels)",
                    tooltip="Number of pixels to feather the mask edges for smooth transitions"
                ),
                # Dilation option for mask
                IO.Int.Input(
                    "mask_dilation", 
                    default=0, 
                    min=-100, 
                    max=100, 
                    step=1,
                    display_name="Mask Dilation (-/+ pixels)",
                    tooltip="Positive values dilate (expand) the mask, negative values erode (shrink) the mask"
                ),
                # RGB color controls with preset helper
                IO.Combo.Input(
                    "color_preset",
                    options=list(cls.PRESETS.keys()) + ["custom"],
                    default="white",
                    display_name="Color Preset",
                    tooltip="Quickly set RGB values. Select 'custom' for manual control."
                ),
                IO.Int.Input("pad_r", default=255, min=0, max=255, step=1, display_name="Padding R (0-255)"),
                IO.Int.Input("pad_g", default=255, min=0, max=255, step=1, display_name="Padding G (0-255)"),
                IO.Int.Input("pad_b", default=255, min=0, max=255, step=1, display_name="Padding B (0-255)"),
            ],
            outputs=[
                IO.Image.Output(),
                IO.Mask.Output("mask")],
        )

    @classmethod
    def execute(
        cls,
        image,
        target_width,
        target_height,
        aspect_ratio_mode,
        content_scale,
        padding_position,
        interpolation,
        feather_pixels,
        mask_dilation,
        color_preset,
        pad_r,
        pad_g,
        pad_b,
    ) -> IO.NodeOutput:
        batch_size, orig_height, orig_width, channels = image.shape
        
        # Apply preset if not "custom" - this sets RGB values BEFORE processing
        if color_preset != "custom" and color_preset in cls.PRESETS:
            pad_r, pad_g, pad_b = cls.PRESETS[color_preset]
        
        # Convert RGB 0-255 → float 0.0-1.0 for tensor operations
        pad_r_f = pad_r / 255.0
        pad_g_f = pad_g / 255.0
        pad_b_f = pad_b / 255.0
        
        # Handle stretch mode (ignore aspect ratio)
        if aspect_ratio_mode == "stretch":
            image_permuted = image.permute(0, 3, 1, 2)
            resized = comfy.utils.common_upscale(
                image_permuted, target_width, target_height, interpolation, "disabled"
            )
            output = resized.permute(0, 2, 3, 1)
            
            # Create mask for stretched image with optional feathering and dilation
            mask = torch.ones((batch_size, 1, target_height, target_width), 
                             dtype=image.dtype, device=image.device)
            
            if mask_dilation != 0:
                mask = cls._apply_mask_dilation(mask, mask_dilation)
            
            if feather_pixels > 0:
                # Apply feathering to edges
                mask = cls._apply_feathering(mask, feather_pixels)
            
            return IO.NodeOutput(output, mask.squeeze(1))
        
        # Calculate scale to preserve aspect ratio
        scale_w = target_width / orig_width
        scale_h = target_height / orig_height
        
        if aspect_ratio_mode == "pad":
            base_scale = min(scale_w, scale_h)  # Fit entire image inside target
        else:  # crop mode
            base_scale = max(scale_w, scale_h)  # Cover entire target area (crop overflow)
        
        # Apply content scale AFTER aspect-ratio-preserving resize
        final_scale = base_scale * content_scale
        new_width = max(1, int(orig_width * final_scale))
        new_height = max(1, int(orig_height * final_scale))
        
        # Resize the image with final scaled dimensions
        image_permuted = image.permute(0, 3, 1, 2)
        resized = comfy.utils.common_upscale(
            image_permuted, new_width, new_height, interpolation, "disabled"
        )
        
        # Create padding canvas with RGB color
        pad_color = [pad_r_f, pad_g_f, pad_b_f]
        if channels == 4:  # Add alpha channel (fully opaque)
            pad_color.append(1.0)
        
        padded = torch.zeros(
            (batch_size, channels, target_height, target_width),
            dtype=image.dtype,
            device=image.device
        )
        
        # Fill canvas with padding color
        for c in range(min(channels, len(pad_color))):
            padded[:, c, :, :] = pad_color[c]
        
        # Calculate padding/cropping offsets based on scaled size
        if aspect_ratio_mode == "pad":
            pad_w = max(0, target_width - new_width)
            pad_h = max(0, target_height - new_height)
        else:  # crop mode
            pad_w = target_width - new_width  # Negative = crop needed
            pad_h = target_height - new_height
        
        # Determine offsets based on position
        if padding_position == "center":
            x_offset = max(0, pad_w // 2)
            y_offset = max(0, pad_h // 2)
        elif padding_position == "top-left":
            x_offset = 0
            y_offset = 0
        elif padding_position == "top-right":
            x_offset = max(0, pad_w)
            y_offset = 0
        elif padding_position == "bottom-left":
            x_offset = 0
            y_offset = max(0, pad_h)
        elif padding_position == "bottom-right":
            x_offset = max(0, pad_w)
            y_offset = max(0, pad_h)
        elif padding_position == "center-top":
            x_offset = max(0, pad_w // 2)
            y_offset = 0
        elif padding_position == "center-bottom":
            x_offset = max(0, pad_w // 2)
            y_offset = max(0, pad_h)
        else:  # Fallback
            x_offset = max(0, pad_w // 2)
            y_offset = max(0, pad_h // 2)
        
        # Handle crop mode (negative offsets = source cropping)
        src_x = max(0, -x_offset)
        src_y = max(0, -y_offset)
        dst_x = max(0, x_offset)
        dst_y = max(0, y_offset)
        
        copy_width = min(new_width - src_x, target_width - dst_x)
        copy_height = min(new_height - src_y, target_height - dst_y)
        
        # Copy resized image into padded canvas
        padded[
            :, :, 
            dst_y:dst_y + copy_height, 
            dst_x:dst_x + copy_width
        ] = resized[
            :, :, 
            src_y:src_y + copy_height, 
            src_x:src_x + copy_width
        ]
        
        # Convert back to BHWC format
        output = padded.permute(0, 2, 3, 1)
        
        # Create mask: 1.0 for image area, 0.0 for padding area (including extra padding from content_scale < 1.0)
        mask = torch.zeros((batch_size, 1, target_height, target_width), 
                          dtype=image.dtype, device=image.device)
        mask[:, 0, dst_y:dst_y + copy_height, dst_x:dst_x + copy_width] = 1.0
        
        # Apply dilation/erosion if specified
        if mask_dilation != 0:
            mask = cls._apply_mask_dilation(mask, mask_dilation)
        
        # Apply feathering if specified - FIXED TO PRESERVE DIMENSIONS
        if feather_pixels > 0:
            mask = cls._apply_feathering(mask, feather_pixels)
        
        return IO.NodeOutput(output, mask.squeeze(1))

    @staticmethod
    def _apply_mask_dilation(mask, dilation_pixels):
        """
        Apply dilation or erosion to mask using OpenCV operations.
        Positive values dilate (expand), negative values erode (shrink).
        """
        if dilation_pixels == 0:
            return mask
        
        device = mask.device
        batch_size, channels, height, width = mask.shape
        
        # Convert tensor to numpy for OpenCV processing
        mask_np = mask.cpu().numpy()
        
        # Create structuring element
        abs_dilation = abs(dilation_pixels)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (abs_dilation * 2 + 1, abs_dilation * 2 + 1))
        
        processed_masks = []
        
        for i in range(batch_size):
            # Get single mask from batch
            single_mask = mask_np[i, 0]  # Shape: (height, width)
            
            # Convert to uint8 for OpenCV
            mask_uint8 = (single_mask * 255).astype(np.uint8)
            
            if dilation_pixels > 0:
                # Dilate (expand)
                processed = cv2.dilate(mask_uint8, kernel, iterations=1)
            else:
                # Erode (shrink)
                processed = cv2.erode(mask_uint8, kernel, iterations=1)
            
            # Convert back to float [0,1]
            processed_float = processed.astype(np.float32) / 255.0
            processed_masks.append(processed_float)
        
        # Stack processed masks back together
        result = np.stack(processed_masks, axis=0)
        result_tensor = torch.from_numpy(result).to(device)
        
        # Reshape to match expected format
        return result_tensor.unsqueeze(1)  # Add channel dimension back

    @staticmethod
    def _apply_feathering(mask, feather_pixels):
        """
        Apply feathering to mask edges using Gaussian blur while PRESERVING EXACT DIMENSIONS.
        Uses proper SAME padding in conv2d to maintain resolution.
        """
        if feather_pixels <= 0:
            return mask
        
        device = mask.device
        batch_size, channels, height, width = mask.shape
        
        # Limit kernel size to prevent artifacts on small images
        kernel_size = min(feather_pixels * 2 + 1, min(height, width))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size for symmetric padding
        
        sigma = max(0.1, feather_pixels * 0.5)
        
        # Create 1D Gaussian kernel
        ax = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        gauss = torch.exp(-ax.pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        
        # Create separable kernels for horizontal and vertical blur
        kernel_h = gauss.view(1, 1, 1, kernel_size)  # [out_ch, in_ch, kH, kW] - horizontal blur
        kernel_v = gauss.view(1, 1, kernel_size, 1)  # vertical blur
        
        # Apply horizontal blur with SAME padding (preserves width)
        # padding=(top/bottom, left/right) → (0, kernel_size//2) for horizontal blur
        blurred = torch.nn.functional.conv2d(
            mask, 
            kernel_h, 
            padding=(0, kernel_size // 2),
            groups=1
        )
        
        # Apply vertical blur with SAME padding (preserves height)
        # padding=(kernel_size//2, 0) for vertical blur
        blurred = torch.nn.functional.conv2d(
            blurred, 
            kernel_v, 
            padding=(kernel_size // 2, 0),
            groups=1
        )
        
        # Safety check: ensure output dimensions match input exactly
        if blurred.shape != mask.shape:
            # Fallback resize (should never trigger with proper padding)
            blurred = torch.nn.functional.interpolate(
                blurred, size=(height, width), mode='bilinear', align_corners=False
            )
        
        return blurred


class CustomImageNodesExtension(ComfyExtension):
    """Extension registering custom image transformation nodes"""
    
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TenguPad,
        ]


async def comfy_entrypoint() -> CustomImageNodesExtension:
    return CustomImageNodesExtension()