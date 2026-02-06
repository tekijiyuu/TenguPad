from __future__ import annotations

import torch
import comfy.utils
from comfy_api.latest import IO, ComfyExtension
from typing_extensions import override


class TenguPad(IO.ComfyNode):
    """
    Advanced image resizing with flexible padding/cropping options and RGB color control.
    Preserves aspect ratio with customizable padding position, true RGB padding color (0-255),
    and multiple aspect ratio modes.
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
            display_name="TenguPad",
            search_aliases=["fit image", "letterbox", "rgb pad", "aspect ratio resize"],
            category="image/transform",
            description=(
                "Resize image while preserving aspect ratio, then pad/crop to target dimensions.\n"
                "Features true RGB padding color control (0-255), customizable position, and aspect modes."
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
                IO.Combo.Input(
                    "padding_position",
                    options=["center", "top-left", "top-right", "bottom-left", "bottom-right"],
                    default="center",
                    tooltip="Position of content within padded canvas"
                ),
                IO.Combo.Input(
                    "interpolation",
                    options=["lanczos", "bicubic", "bilinear", "area", "nearest-exact"],
                    default="lanczos"
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
                IO.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        image,
        target_width,
        target_height,
        aspect_ratio_mode,
        padding_position,
        interpolation,
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
            
           
            return IO.NodeOutput(output)
        
        # Calculate scale to preserve aspect ratio
        scale_w = target_width / orig_width
        scale_h = target_height / orig_height
        
        if aspect_ratio_mode == "pad":
            scale = min(scale_w, scale_h)  # Fit entire image inside target
        else:  # crop mode
            scale = max(scale_w, scale_h)  # Cover entire target area (crop overflow)
        
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize the image
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
        
        # Calculate padding/cropping offsets
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
        
        # Calculate actual padding values for optional output
        pad_left = dst_x
        pad_top = dst_y
        pad_right = target_width - (dst_x + copy_width)
        pad_bottom = target_height - (dst_y + copy_height)
        
        
        return IO.NodeOutput(output)


class CustomImageNodesExtension(ComfyExtension):
    """Extension registering custom image transformation nodes"""
    
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TenguPad,
        ]


async def comfy_entrypoint() -> CustomImageNodesExtension:
    return CustomImageNodesExtension()