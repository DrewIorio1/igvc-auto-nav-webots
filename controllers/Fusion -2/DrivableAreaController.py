"""
    File: DrivableAreaController.py - A controller for detecting drivable areas in images using YOLOPv2.

    This module provides functionality to process images from a camera, detect drivable areas,
    and optionally overlay the detected areas on the original image. It uses a YOLOPv2 model for
    semantic segmentation to identify drivable areas and lane lines in the input images.

    Required Libraries:
       cv2 - For image processing and manipulation.
       numpy - For numerical operations and array manipulations.
       torch - For loading and running the YOLOPv2 model.

    Classes:
        DrivableAreaDetector - A class that processes images to detect drivable areas using YOLOPv2.
        Functions:
            - __init__: Initializes the detector with camera parameters and model settings.
            - _init_model: Loads the YOLOPv2 model from a file.
            - letterbox: Performs letterboxing on the input image to fit it into the model's input shape.
            - _preprocess: Converts the input image to RGB, applies letterboxing, and converts it to a tensor.
            - driving_area_mask: Processes segmentation output to extract drivable area mask.
            - lane_line_mask: Processes lane line segmentation output to extract lane line mask.
            - process_frame: Processes a single frame from the camera buffer to detect drivable areas.
"""

import cv2
import numpy as np
import torch

class DrivableAreaDetector:
    """
        DrivableAreaDetector is a class that processes images from a camera to detect drivable areas
        using a YOLOPv2 model. It performs letterboxing, preprocessing, and segmentation to identify
        drivable areas and lane lines in the input images. The detected areas can be overlaid on the
        original image for visualization.

        Variables:
            - camera: The camera object providing the input images.
            - original_h: The height of the original image.
            - original_w: The width of the original image.
            - model_input_h: The height of the input image for the model.
            - model_input_w: The width of the input image for the model.
            - device: The device on which the model runs (e.g., 'cpu' or 'cuda').
            - stride: The stride used for letterboxing.
            
    """
    def __init__(self, camera, original_h, original_w,
                 model_input_h=384, model_input_w=640,
                 device='cpu', stride=32):
        self.camera = camera
        self.original_h = original_h
        self.original_w = original_w
        self.model_input_h = model_input_h
        self.model_input_w = model_input_w
        self.device = device
        self.stride = stride

        # Load YOLOPv2 model
        self.model = self._init_model()
        self.last_letterbox_info = None

    def _init_model(self):
        """
            Initializes the YOLOPv2 model by loading it from a file.
            Returns the loaded model.
        """
        try:
            model = torch.load("yolopv2.pt", map_location=self.device)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading YOLOPv2 model: {e}")
            raise

    def letterbox(self, img, new_shape=(384, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        """
            Performs letterboxing on the input image to fit it into the specified new shape.

            Args:
                img (numpy.ndarray): The input image to be letterboxed.
                new_shape (tuple): The desired shape for the output image.
                color (tuple): The color used for padding.
                auto (bool): Whether to automatically adjust padding.
                scaleFill (bool): Whether to fill the entire new shape without maintaining aspect ratio.
                scaleup (bool): Whether to allow scaling up of the image.
        """
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
        dw /= 2
        dh /= 2
        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color
        )
        # store letterbox info
        self.last_letterbox_info = {
            'pad': (dw * 2, dh * 2),
            'letterboxed_shape': img_padded.shape[:2]
        }
        return img_padded

    def _preprocess(self, img_np_bgra):
        """
            Preprocesses the input image by converting it to RGB, performing letterboxing, 
            and converting it to a tensor suitable for the model.

            Args:
            img_np_bgra (numpy.ndarray): The input image in BGRA format.
            Returns:
                torch.Tensor: The preprocessed image tensor ready for model input.
        """
        img_rgb = cv2.cvtColor(img_np_bgra, cv2.COLOR_BGRA2RGB)
        img_lb = self.letterbox(img_rgb, new_shape=(self.model_input_h, self.model_input_w))
        img_tensor = torch.from_numpy(img_lb).float().permute(2, 0, 1)[None] / 255.0
        return img_tensor.to(self.device)

    def driving_area_mask(self, seg_output):
        """
            Processes the segmentation output to extract the drivable area mask.

            Args:
                seg_output (torch.Tensor): The output from the segmentation model.
        """
        _, da_seg_mask = torch.max(seg_output, 1)
        da_mask_np = da_seg_mask.squeeze().cpu().numpy().astype(np.uint8)
        return da_mask_np

    def lane_line_mask(self, ll_output):
        """
            Processes the lane line output to extract the lane line mask.

            Args:
            ll_output (torch.Tensor): The output from the lane line segmentation model.

            Returns:
            numpy.ndarray: The processed lane line mask as a binary image.

        """
        #ll_seg_mask = torch.sigmoid(ll_output)
        #ll_seg_mask = torch.round(ll_seg_mask).squeeze(1)
        #ll_mask_np = ll_seg_mask.cpu().numpy().astype(np.uint8)
        #return ll_mask_np
        _, ll_seg_mask = torch.max(ll_output, 1)
        ll_seg_mask = ll_seg_mask.squeeze().cpu().numpy().astype(np.uint8)
        return ll_seg_mask


    def undo_letterbox_and_resize(self, mask, letterbox_info, target_size):
        # 1) Ensure mask is plain 2D numpy
        if hasattr(mask, 'cpu'):    # torch tensor?
            mask = mask.cpu().detach().numpy()
        mask = np.squeeze(mask)     # drop any 1−length dims

        # If it’s H×W×C, reduce to single channel
        if mask.ndim == 3:
            mask = mask[..., 0]

        # Debug print
        print(f"[UNDO] mask.shape={mask.shape}, "
              f"letterbox={letterbox_info}, "
              f"target={target_size}")

        # 2) Extract dims
        pad_w, pad_h = map(int, letterbox_info['pad'])
        H_l, W_l     = map(int, letterbox_info['letterboxed_shape'])
        W_t, H_t     = map(int, target_size)

        # 3) Compute content box
        content_h = H_l - pad_h
        content_w = W_l - pad_w
        top       = pad_h // 2
        left      = pad_w // 2
        bottom    = top + content_h
        right     = left + content_w

        # Clamp to valid indices
        top, bottom = np.clip([top, bottom], 0, H_l)
        left, right = np.clip([left, right], 0, W_l)

        cropped = mask[top:bottom, left:right]
        if cropped.size == 0:
            raise ValueError(
                f"Empty crop after squeeze! mask.shape={mask.shape}, "
                f"crop=[{top}:{bottom},{left}:{right}]"
            )

        # 4) Resize back
        return cv2.resize(cropped * 255,
                          (W_t, H_t),
                          interpolation=cv2.INTER_NEAREST)
            

    def process_frame(self, buf, display_overlay=True, overlay_alpha=0.7):
        """
            Processes a single frame from the camera buffer to detect drivable areas.

            Args:
                buf (bytes): The camera buffer containing the image data.
                display_overlay (bool): Whether to display the overlay of detected areas.
                overlay_alpha (float): The alpha value for the overlay transparency.

            Returns:
                numpy.ndarray: The processed mask of drivable areas.
        """
        img_np_bgra = np.frombuffer(buf, np.uint8).reshape((self.original_h, self.original_w, 4))
        original_rgb = cv2.cvtColor(img_np_bgra, cv2.COLOR_BGRA2RGB) if display_overlay else None

        tensor = self._preprocess(img_np_bgra)
        with torch.no_grad():
            outputs = self.model(tensor)
            
        """
        da_mask_model_res = self.driving_area_mask(outputs[1])
        ll_mask_model_res = self.lane_line_mask(outputs[2])
        # Use only drivable-area mask
        combined_mask_letterboxed = da_mask_model_res

        # Remove letterbox padding
        pad_w, pad_h = self.last_letterbox_info['pad']
        letterboxed_h, letterboxed_w = self.last_letterbox_info['letterboxed_shape']
        top_pad_pixels  = int(round(pad_h / 2 - 0.1))
        left_pad_pixels = int(round(pad_w / 2 - 0.1))
        content_h = letterboxed_h - int(round(pad_h))
        content_w = letterboxed_w - int(round(pad_w))
        mask_no_padding = combined_mask_letterboxed[
            top_pad_pixels : top_pad_pixels + content_h,
            left_pad_pixels: left_pad_pixels + content_w
        ]

        # Resize to original resolution
        mask_full = cv2.resize(mask_no_padding * 255,
                               (self.original_w, self.original_h),
                               interpolation=cv2.INTER_NEAREST)
        
        combined_mask_letterboxed = ll_mask_model_res

        # Remove letterbox padding
        pad_w, pad_h = self.last_letterbox_info['pad']
        letterboxed_h, letterboxed_w = self.last_letterbox_info['letterboxed_shape']
        top_pad_pixels  = int(round(pad_h / 2 - 0.1))
        left_pad_pixels = int(round(pad_w / 2 - 0.1))
        content_h = letterboxed_h - int(round(pad_h))
        content_w = letterboxed_w - int(round(pad_w))
        mask_no_padding = combined_mask_letterboxed[
            top_pad_pixels : top_pad_pixels + content_h,
            left_pad_pixels: left_pad_pixels + content_w
        ]

        # Resize to original resolution
        mask_full_lane = cv2.resize(mask_no_padding * 255,
                               (self.original_w, self.original_h),
                               interpolation=cv2.INTER_NEAREST)
        
        """
        da_mask_lb = self.driving_area_mask(outputs[1])
        ll_mask_lb = self.lane_line_mask     (outputs[2])
        # 2) Prepare args
        letterbox_info = self.last_letterbox_info
        size           = (self.original_w, self.original_h)
        
        # 3) Undo letterbox + resize for both
        mask_full      = self.undo_letterbox_and_resize(da_mask_lb,
                                                   letterbox_info,
                                                   size)
        
        mask_full_lane = self.undo_letterbox_and_resize(ll_mask_lb,
                                                   letterbox_info,
                                                   size)
       
       
        if display_overlay and original_rgb is not None:
            colored = np.zeros_like(original_rgb)
            colored[mask_full == 255] = (0, 255, 0)
            overlayed = cv2.addWeighted(original_rgb, 1 - overlay_alpha, colored, overlay_alpha, 0)
            cv2.imshow('Drivable area', overlayed)

        return mask_full
        
     
    

    