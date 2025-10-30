import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torchvision.transforms as transforms

__all__ = ['YOLOInference', 'Results', 'YOLOv8Model']


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Detect(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat[:, :self.reg_max * 4], x_cat[:, self.reg_max * 4:]
            dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1) if self.export else (torch.cat((dbox, cls.sigmoid()), 1), x)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


class YOLOv8Model(nn.Module):
    def __init__(self, cfg=None, ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = {}
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml = {'nc': nc or 80, 'scales': {'n': [0.25, 0.25], 'm': [0.67, 0.75]}}
        
        self.model = self._build_model()
        self.stride = torch.tensor([8., 16., 32.])
        self.names = {i: f'class{i}' for i in range(self.yaml['nc'])}

    def _build_model(self):
        layers = []
        
        layers.extend([
            Conv(3, 48, 3, 2),
            Conv(48, 96, 3, 2),
            C2f(96, 96, 2, True),
            Conv(96, 192, 3, 2),
            C2f(192, 192, 4, True),
            Conv(192, 384, 3, 2),
            C2f(384, 384, 4, True),
            Conv(384, 576, 3, 2),
            C2f(576, 576, 2, True),
            SPPF(576, 576, 5),
        ])
        
        layers.extend([
            nn.Upsample(None, 2, 'nearest'),
            Conv(576, 384, 1, 1),
            C2f(768, 384, 2),
            nn.Upsample(None, 2, 'nearest'),
            Conv(384, 192, 1, 1),
            C2f(384, 192, 2),
            Conv(192, 192, 3, 2),
            C2f(576, 384, 2),
            Conv(384, 384, 3, 2),
            C2f(960, 576, 2),
            Detect(self.yaml['nc'], (192, 384, 576)),
        ])
        
        return nn.ModuleList(layers)

    def forward(self, x):
        y = []
        for i, m in enumerate(self.model):
            if i in [10, 13]:
                x = m(x)
                x = torch.cat([x, y[i - 4]], 1) if i == 10 else torch.cat([x, y[i - 6]], 1)
            elif i in [11, 14, 16, 18]:
                if i in [11, 14]:
                    x = m(x)
                else:
                    x = m(x)
                    x = torch.cat([x, y[i - 4]], 1)
            else:
                x = m(x)
            y.append(x if i in [4, 6, 9, 12, 15, 17, 19] else None)
        
        return x

    def fuse(self):
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        return self


def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         groups=conv.groups,
                         bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class YOLOInference:
    def __init__(self, model_path: str, device: str = 'auto', conf_thresh: float = 0.25):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        self.conf_thresh = conf_thresh
        self.model = self._load_model(model_path)
        self.model.eval()
        
        self.stream = torch.cuda.current_stream() if torch.cuda.is_available() else None
        self.tensor_cache = {}
        self._warmup()
    
    def _load_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if 'model' in checkpoint:
                model = checkpoint['model']
                if hasattr(model, 'float'):
                    model = model.float()
            else:
                model = YOLOv8Model()
                if isinstance(checkpoint, dict):
                    model.load_state_dict(checkpoint, strict=False)
        except Exception:
            model = YOLOv8Model()
            
        model = model.to(self.device)
        model.eval()
        
        if torch.cuda.is_available():
            try:
                model = torch.jit.script(model)
            except Exception:
                pass
        
        return model
    
    def _warmup(self):
        dummy_input = torch.zeros(1, 3, 640, 640, device=self.device)
        with torch.no_grad():
            with torch.cuda.device(self.device):
                for _ in range(3):
                    self.model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def preprocess(self, image: np.ndarray, target_size: int = 640) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        top = (target_size - new_h) // 2
        bottom = target_size - new_h - top
        left = (target_size - new_w) // 2
        right = target_size - new_w - left
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        key = (target_size, target_size)
        if key not in self.tensor_cache:
            self.tensor_cache[key] = torch.zeros((1, 3, target_size, target_size), device=self.device, dtype=torch.float32)
        
        tensor = self.tensor_cache[key]
        tensor[0] = torch.from_numpy(padded.transpose(2, 0, 1)).float().to(self.device) / 255.0
        
        return tensor, scale, (left, top)
    
    def postprocess(self, predictions: torch.Tensor, scale: float, offset: Tuple[int, int], 
                   original_shape: Tuple[int, int]) -> List[Dict]:
        if isinstance(predictions, tuple):
            pred = predictions[0]
        else:
            pred = predictions
        
        if pred.dim() == 3:
            pred = pred.squeeze(0)
        
        pred = pred.transpose(0, 1)
        
        boxes = pred[:, :4]
        scores = pred[:, 4:].max(dim=1)[0]
        classes = pred[:, 4:].argmax(dim=1)
        
        mask = scores > self.conf_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]
        
        if len(boxes) == 0:
            return []
        
        boxes_xyxy = torch.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        
        keep = self._nms(boxes_xyxy, scores, iou_threshold=0.45)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        classes = classes[keep]
        
        detections = []
        left, top = offset
        
        for box, score, cls in zip(boxes_xyxy, scores, classes):
            x1, y1, x2, y2 = box
            
            x1 = (x1 - left) / scale
            y1 = (y1 - top) / scale
            x2 = (x2 - left) / scale
            y2 = (y2 - top) / scale
            
            x1 = max(0, min(x1, original_shape[1]))
            y1 = max(0, min(y1, original_shape[0]))
            x2 = max(0, min(x2, original_shape[1]))
            y2 = max(0, min(y2, original_shape[0]))
            
            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'conf': float(score),
                'cls': int(cls)
            })
        
        return detections
    
    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45) -> torch.Tensor:
        return torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    
    def detect(self, image: np.ndarray, classes: Optional[List[int]] = None) -> List[Dict]:
        original_shape = image.shape[:2]
        
        tensor, scale, offset = self.preprocess(image)
        
        with torch.no_grad():
            with torch.cuda.device(self.device):
                predictions = self.model(tensor)
        
        detections = self.postprocess(predictions, scale, offset, original_shape)
        
        if classes is not None:
            detections = [det for det in detections if det['cls'] in classes]
        
        return detections


class Results:
    def __init__(self, detections: List[Dict]):
        self.detections = detections
        self._boxes = None
        self._conf = None
        self._cls = None
        self._xywh = None
    
    @property
    def boxes(self):
        if self._boxes is None:
            self._boxes = torch.tensor([det['bbox'] for det in self.detections])
        return self._boxes
    
    @property
    def conf(self):
        if self._conf is None:
            self._conf = torch.tensor([det['conf'] for det in self.detections])
        return self._conf
    
    @property
    def cls(self):
        if self._cls is None:
            self._cls = torch.tensor([det['cls'] for det in self.detections])
        return self._cls
    
    @property
    def xywh(self):
        if self._xywh is None:
            boxes = []
            for det in self.detections:
                x1, y1, x2, y2 = det['bbox']
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                boxes.append([x, y, w, h])
            self._xywh = torch.tensor(boxes)
        return self._xywh
    
    def __len__(self):
        return len(self.detections)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return Results(self.detections[index])
        return Results([self.detections[index]])