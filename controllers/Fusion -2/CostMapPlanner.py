"""

History 
     5/10/2025 AI - Added Yolop_pretrained.pt model & lib folder to run
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# ——— Model & Preprocessing Setup ———
MODEL_WEIGHTS = "./yolop_pretrained.pt"
transform = transforms.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = torch.load(MODEL_WEIGHTS, map_location=device)
#model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
model = model.to(device).eval()

lane_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
SMOOTH_ALPHA = 0.8
_prev_cost_map = None

def pad_to_stride(img, stride=32):
    """Pad a H×W×C numpy image so H and W are multiples of stride."""
    h, w = img.shape[:2]
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=0)
    return padded, (top, bottom, left, right)

def unpad_mask(mask, pads):
    top, bottom, left, right = pads
    if bottom == 0:
        mask = mask[top:, :]
    else:
        mask = mask[top:-bottom, :]
    if right == 0:
        mask = mask[:, left:]
    else:
        mask = mask[:, left:-right]
    return mask

def run_yolop_segmentation(frame_bgr):
    H, W = frame_bgr.shape[:2]

    # 1) pad so dimensions are multiples of 32
    padded, pads = pad_to_stride(frame_bgr, stride=32)
    ph, pw = padded.shape[:2]

    # 2) to RGB and tensor
    frame_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    img = transform(frame_rgb).unsqueeze(0).to(device)

    # 3) forward
    with torch.no_grad():
        outputs = model(img)
    # YOLOP sometimes returns det_out as a Tensor, sometimes as a tuple/list.
    det_out_raw, da_seg_out, ll_seg_out = outputs

    # Normalize det_out into a Python list of boxes
    if isinstance(det_out_raw, torch.Tensor):
        det_out = det_out_raw.cpu().numpy().tolist()
    else:
        # assume it’s already a sequence of boxes
        det_out = list(det_out_raw)

    # upsample and unpad as before
    da_up = F.interpolate(da_seg_out, size=(ph, pw),
                         mode='bilinear', align_corners=False)
    ll_up = F.interpolate(ll_seg_out, size=(ph, pw),
                         mode='bilinear', align_corners=False)

    da_pred = torch.argmax(da_up, dim=1)[0].cpu().numpy().astype(np.uint8)
    ll_pred = torch.argmax(ll_up, dim=1)[0].cpu().numpy().astype(np.uint8)

    drivable_mask = unpad_mask(da_pred, pads)
    lane_mask    = unpad_mask(ll_pred, pads)

    # Sanity‐check sizing
    assert drivable_mask.shape == (H, W)
    assert lane_mask.shape    == (H, W)

    return det_out, drivable_mask, lane_mask

def update_cost_map_old(frame_bgr):
    global _prev_cost_map
    H, W = frame_bgr.shape[:2]

    det_out, drivable_mask, lane_mask = run_yolop_segmentation(frame_bgr)

    # combine drivable+lanes
    lane_dilated = cv2.dilate(lane_mask, lane_kernel, iterations=2)
    valid = (drivable_mask & lane_dilated).astype(np.uint8)

    # soft cost
    dist = cv2.distanceTransform(valid, cv2.DIST_L2, 5)
    if dist.max() > 0:
        base = 1.0 - (dist / dist.max())
    else:
        base = np.ones_like(dist, dtype=np.float32)

    # overlay detections
    cost = base.copy().astype(np.float32)
    for det in det_out:
        if len(det) == 6:
            x1, y1, x2, y2, score, cls = det
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(cost, (x1, y1), (x2, y2), color=1.0, thickness=-1)
        #else:
        #    print(f"[Warning] Unexpected detection format: {det}")
    
    cost = cv2.GaussianBlur(cost, (11,11), 0)

    # temporal smoothing
    if _prev_cost_map is None:
        _prev_cost_map = cost
    cost = SMOOTH_ALPHA * _prev_cost_map + (1-SMOOTH_ALPHA) * cost
    _prev_cost_map = cost

    return cost

def update_cost_map(frame_bgr, cost_threshold=0.3):
    """
    Runs YOLOP, builds a soft cost map, AND returns a binary obstacle map
      - cost_threshold: float in [0,1]; cost > threshold → obstacle
    """
    global _prev_cost_map

    # 1) original soft‐cost pipeline
    det_out, drivable_mask, lane_mask = run_yolop_segmentation(frame_bgr)
    lane_dilated = cv2.dilate(lane_mask, lane_kernel, iterations=2)
    valid = (drivable_mask & lane_dilated).astype(np.uint8)

    dist = cv2.distanceTransform(valid, cv2.DIST_L2, 5)
    base = 1.0 - (dist / dist.max()) if dist.max()>0 else np.ones_like(dist, np.float32)

    cost = base.copy().astype(np.float32)
    for det in det_out:
        if len(det)==6:
            x1,y1,x2,y2,_,_ = map(int,det)
            cv2.rectangle(cost, (x1,y1), (x2,y2), color=1.0, thickness=-1)

    cost = cv2.GaussianBlur(cost, (11,11), 0)
    if _prev_cost_map is None:
        _prev_cost_map = cost
    cost = SMOOTH_ALPHA*_prev_cost_map + (1-SMOOTH_ALPHA)*cost
    _prev_cost_map = cost

    # 2) threshold → binary occupancy
    #    any pixel with cost > threshold becomes obstacle (1), else free (0)
    obstacle_map = (cost > cost_threshold).astype(np.uint8)
    drivable_area = (obstacle_map == 0).astype(np.uint8)  # 1 where drivable, 0 
    return cost, obstacle_map, drivable_area, drivable_mask



__all__ = ["run_yolop_segmentation", "update_cost_map"]
