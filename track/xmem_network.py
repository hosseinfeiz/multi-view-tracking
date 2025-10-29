import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class EnhancedFeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()
        total_input_channels = x_in_dim + g_in_dim
        self.conv1 = nn.Conv2d(total_input_channels, g_mid_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(g_mid_dim, g_out_dim, 3, padding=1)
        self.attention = nn.Conv2d(g_mid_dim, g_mid_dim, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x, g):
        batch_size, num_objects = g.shape[:2]
        x_expanded = x.unsqueeze(1).expand(-1, num_objects, -1, -1, -1)
        x_flat = x_expanded.flatten(start_dim=0, end_dim=1)
        g_flat = g.flatten(start_dim=0, end_dim=1)
        
        actual_channels = x_flat.shape[1] + g_flat.shape[1]
        expected_channels = self.conv1.in_channels
        
        if actual_channels != expected_channels:
            if g_flat.shape[1] > expected_channels - x_flat.shape[1]:
                target_g_channels = expected_channels - x_flat.shape[1]
                g_flat = g_flat[:, :target_g_channels]
            elif g_flat.shape[1] < expected_channels - x_flat.shape[1]:
                target_g_channels = expected_channels - x_flat.shape[1]
                padding = torch.zeros(g_flat.shape[0], target_g_channels - g_flat.shape[1], 
                                    g_flat.shape[2], g_flat.shape[3], device=g_flat.device)
                g_flat = torch.cat([g_flat, padding], dim=1)
        
        combined = torch.cat([x_flat, g_flat], dim=1)
        out = self.relu(self.conv1(combined))
        
        attention_weights = torch.sigmoid(self.attention(out))
        out = out * attention_weights
        
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        return out.view(batch_size, num_objects, *out.shape[1:])

class OptimizedResNetBackbone(nn.Module):
    def __init__(self, layers=[2, 2, 3, 2], width_multiplier=1.0):
        super().__init__()
        base_width = int(32 * width_multiplier)
        
        self.conv1 = nn.Conv2d(3, base_width, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(base_width, base_width * 4, layers[0], stride=1)
        self.layer2 = self._make_layer(base_width * 4, base_width * 8, layers[1], stride=2)
        self.layer3 = self._make_layer(base_width * 8, base_width * 16, layers[2], stride=2)
        
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        f4 = self.layer1(x)
        f8 = self.layer2(f4)
        f16 = self.layer3(f8)
        
        return f16, f8, f4

class KeyEncoder(nn.Module):
    def __init__(self, width_multiplier=1.0):
        super().__init__()
        self.backbone = OptimizedResNetBackbone(width_multiplier=width_multiplier)

    def forward(self, f):
        return self.backbone(f)

class EnhancedValueEncoder(nn.Module):
    def __init__(self, value_dim, hidden_dim, single_object=False):
        super().__init__()
        self.single_object = single_object
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        
        input_dim = 3 + (1 if single_object else 2)
        self.backbone = OptimizedResNetBackbone(width_multiplier=0.75)
        self.backbone.conv1 = nn.Conv2d(input_dim, int(32 * 0.75), 7, stride=2, padding=3, bias=False)
        
        backbone_out_dim = int(32 * 0.75 * 16)
        self.fuser = EnhancedFeatureFusionBlock(backbone_out_dim, backbone_out_dim, value_dim // 2, value_dim)
        
        if hidden_dim > 0:
            self.hidden_update = nn.Sequential(
                nn.Conv2d(value_dim + hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.Tanh()
            )

    def forward(self, image, image_feat_f16, h, masks, others, is_deep_update=True):
        if not self.single_object:
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        
        batch_size, num_objects = g.shape[:2]
        
        inputs = []
        for obj_idx in range(num_objects):
            if not self.single_object:
                obj_input = torch.cat([image, g[:, obj_idx, :, :, :]], dim=1)
            else:
                obj_input = torch.cat([image, g[:, obj_idx, 0:1, :, :]], dim=1)
            inputs.append(obj_input)
        
        features = []
        for obj_input in inputs:
            f16, f8, f4 = self.backbone(obj_input)
            features.append(f16)
        
        g_features = torch.stack(features, dim=1)
        g = self.fuser(image_feat_f16, g_features)
        
        if is_deep_update and self.hidden_dim > 0 and h is not None:
            g_h, g_w = g.shape[-2:]
            h_h, h_w = h.shape[-2:]
            
            if g_h != h_h or g_w != h_w:
                h = F.interpolate(
                    h.flatten(start_dim=0, end_dim=1), 
                    size=(g_h, g_w), 
                    mode='bilinear', 
                    align_corners=False
                ).view(batch_size, num_objects, self.hidden_dim, g_h, g_w)
            
            if h.shape[0] != batch_size or h.shape[1] != num_objects:
                device = g.device
                h = torch.zeros(batch_size, num_objects, self.hidden_dim, g_h, g_w, device=device)
            
            h_flat = h.flatten(start_dim=0, end_dim=1)
            g_flat = g.flatten(start_dim=0, end_dim=1)
            
            if h_flat.shape[1] != self.hidden_dim:
                device = g.device
                h_flat = torch.zeros(g_flat.shape[0], self.hidden_dim, g_h, g_w, device=device)
            
            combined = torch.cat([g_flat, h_flat], dim=1)
            h_new = self.hidden_update(combined)
            h = h_new.view(batch_size, num_objects, self.hidden_dim, g_h, g_w)
        
        return g, h

class EnhancedKeyProjection(nn.Module):
    def __init__(self, in_dim, key_dim):
        super().__init__()
        self.key_proj = nn.Conv2d(in_dim, key_dim, kernel_size=3, padding=1)
        self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        self.e_proj = nn.Conv2d(in_dim, key_dim, kernel_size=3, padding=1)
        
        self.key_norm = nn.GroupNorm(8, key_dim)
        self.e_norm = nn.GroupNorm(8, key_dim)
        
        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x, need_s, need_e):
        key = self.key_norm(self.key_proj(x))
        shrinkage = self.d_proj(x)**2 + 1 if need_s else None
        selection = torch.sigmoid(self.e_norm(self.e_proj(x))) if need_e else None
        return key, shrinkage, selection

class EnhancedUpsampleBlock(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(g_up_dim),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(g_up_dim, g_out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(g_out_dim),
            nn.ReLU(inplace=True)
        )
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_g):
        if skip_f.dim() == 5:
            batch_size, num_objects = skip_f.shape[:2]
            skip_f = skip_f.flatten(start_dim=0, end_dim=1)
        elif skip_f.dim() == 4:
            batch_size, num_objects = up_g.shape[:2]
        else:
            raise ValueError(f"Unexpected skip_f dimensions: {skip_f.shape}")
        
        skip_f = self.skip_conv(skip_f)
        
        g_flat = up_g.flatten(start_dim=0, end_dim=1)
        g_up = F.interpolate(g_flat, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        if skip_f.shape[0] != g_up.shape[0]:
            if skip_f.shape[0] == 1:
                skip_f = skip_f.expand(g_up.shape[0], -1, -1, -1)
            else:
                repeat_factor = g_up.shape[0] // skip_f.shape[0]
                if repeat_factor > 1:
                    skip_f = skip_f.repeat(repeat_factor, 1, 1, 1)
        
        g_combined = g_up + skip_f
        g_out = self.out_conv(g_combined)
        
        return g_out.view(batch_size, num_objects, *g_out.shape[1:])

class EnhancedDecoder(nn.Module):
    def __init__(self, val_dim, hidden_dim):
        super().__init__()
        self.val_dim = val_dim
        self.hidden_dim = hidden_dim
        
        fuser_input_dim = val_dim + hidden_dim if hidden_dim > 0 else val_dim
        backbone_out_dim = int(32 * 16)
        self.fuser = EnhancedFeatureFusionBlock(backbone_out_dim, fuser_input_dim, 256, 256)
        
        if hidden_dim > 0:
            self.hidden_update = nn.Sequential(
                nn.Conv2d(128 + 1, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.Tanh()
            )
        
        self.up_16_8 = EnhancedUpsampleBlock(int(32 * 8), 256, 128)
        self.up_8_4 = EnhancedUpsampleBlock(int(32 * 4), 128, 128)
        
        self.pred = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, f16, f8, f4, hidden_state, memory_readout, h_out=True):
        batch_size, num_objects = memory_readout.shape[:2]
        
        if self.hidden_dim > 0 and hidden_state is not None:
            h_batch, h_objects = hidden_state.shape[:2]
            
            if h_batch != batch_size or h_objects != num_objects:
                if h_objects == num_objects:
                    hidden_state = hidden_state[:batch_size]
                elif h_batch == batch_size:
                    hidden_state = hidden_state[:, :num_objects]
                else:
                    device = hidden_state.device
                    h, w = memory_readout.shape[-2:]
                    hidden_state = torch.zeros(batch_size, num_objects, self.hidden_dim, h, w, device=device)
            
            mem_h, mem_w = memory_readout.shape[-2:]
            hid_h, hid_w = hidden_state.shape[-2:]
            
            if mem_h != hid_h or mem_w != hid_w:
                hidden_flat = hidden_state.flatten(start_dim=0, end_dim=1)
                hidden_resized = F.interpolate(hidden_flat, size=(mem_h, mem_w), mode='bilinear', align_corners=False)
                hidden_state = hidden_resized.view(batch_size, num_objects, self.hidden_dim, mem_h, mem_w)
            
            combined_input = torch.cat([memory_readout, hidden_state], 2)
        else:
            combined_input = memory_readout
        
        g16 = self.fuser(f16, combined_input)
        g8 = self.up_16_8(f8, g16)
        g4 = self.up_8_4(f4, g8)
        
        g4_flat = g4.flatten(start_dim=0, end_dim=1)
        logits = self.pred(g4_flat)
        
        if h_out and self.hidden_dim > 0 and hidden_state is not None:
            logits_expanded = logits.view(batch_size, num_objects, 1, *logits.shape[-2:])
            
            g4_h, g4_w = g4.shape[-2:]
            logits_resized = F.interpolate(
                logits_expanded.flatten(start_dim=0, end_dim=1), 
                size=(g4_h, g4_w), 
                mode='bilinear', 
                align_corners=False
            )
            logits_resized = logits_resized.view(batch_size, num_objects, 1, g4_h, g4_w)
            
            g4_with_logits = torch.cat([
                g4.flatten(start_dim=0, end_dim=1),
                logits_resized.flatten(start_dim=0, end_dim=1)
            ], dim=1)
            
            hidden_state_new = self.hidden_update(g4_with_logits)
            hidden_state = hidden_state_new.view(batch_size, num_objects, self.hidden_dim, *hidden_state_new.shape[-2:])
        else:
            hidden_state = None
        
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])
        
        return hidden_state, logits

class XMem(nn.Module):
    def __init__(self, config, model_path=None, map_location=None):
        super().__init__()
        model_weights = self.init_hyperparameters(config, model_path, map_location)

        self.single_object = config.get('single_object', False)
        width_multiplier = config.get('width_multiplier', 1.0)
        
        self.key_encoder = KeyEncoder(width_multiplier=width_multiplier)
        self.value_encoder = EnhancedValueEncoder(self.value_dim, self.hidden_dim, self.single_object)
        self.key_proj = EnhancedKeyProjection(int(32 * width_multiplier * 16), self.key_dim)
        self.decoder = EnhancedDecoder(self.value_dim, self.hidden_dim)

        if model_weights is not None:
            self.load_weights(model_weights, init_as_zero_if_needed=True)

    def encode_key(self, frame, need_sk=True, need_ek=True):
        if len(frame.shape) == 5:
            need_reshape = True
            b, t = frame.shape[:2]
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            need_reshape = False
        else:
            raise NotImplementedError
    
        f16, f8, f4 = self.key_encoder(frame)
        key, shrinkage, selection = self.key_proj(f16, need_sk, need_ek)

        if need_reshape:
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            if shrinkage is not None:
                shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
            if selection is not None:
                selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()
            f16 = f16.view(b, t, *f16.shape[-3:])
            f8 = f8.view(b, t, *f8.shape[-3:])
            f4 = f4.view(b, t, *f4.shape[-3:])

        return key, shrinkage, selection, f16, f8, f4

    def encode_value(self, frame, image_feat_f16, h16, masks, is_deep_update=True):
        num_objects = masks.shape[1]
        if num_objects != 1:
            others = torch.cat([
                torch.sum(
                    masks[:, [j for j in range(num_objects) if i!=j]]
                , dim=1, keepdim=True)
            for i in range(num_objects)], 1)
        else:
            others = torch.zeros_like(masks)

        g16, h16 = self.value_encoder(frame, image_feat_f16, h16, masks, others, is_deep_update)
        return g16, h16

    def read_memory(self, query_key, query_selection, memory_key, memory_shrinkage, memory_value):
        batch_size, num_objects = memory_value.shape[:2]
        if memory_value.size(-1) > 0:
            memory = torch.mean(memory_value, dim=-1)
        else:
            memory = torch.zeros(batch_size, num_objects, self.value_dim, 
                               query_key.size(-2), query_key.size(-1), 
                               device=query_key.device)
        return memory

    def segment(self, multi_scale_features, memory_readout, hidden_state, selector=None, h_out=True, strip_bg=True):
        hidden_state, logits = self.decoder(*multi_scale_features, hidden_state, memory_readout, h_out=h_out)
        prob = torch.sigmoid(logits)
        
        if selector is not None:
            prob = prob * selector
        
        prob_with_bg = torch.cat([1 - torch.sum(prob, dim=1, keepdim=True), prob], dim=1)
        logits_with_bg = torch.log(prob_with_bg / (1 - prob_with_bg + 1e-8))
        
        if strip_bg:
            prob = prob_with_bg[:, 1:]
        else:
            prob = prob_with_bg

        return hidden_state, logits_with_bg, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'read_memory':
            return self.read_memory(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError

    def init_hyperparameters(self, config, model_path=None, map_location=None):
        if model_path is not None:
            try:
                model_weights = torch.load(model_path, map_location=map_location)
                self.key_dim = model_weights.get('key_proj.key_proj.weight', torch.zeros(64, 512, 3, 3)).shape[0]
                self.value_dim = model_weights.get('value_encoder.fuser.conv2.weight', torch.zeros(256, 256, 3, 3)).shape[0]
                self.disable_hidden = 'decoder.hidden_update.weight' not in model_weights
                if self.disable_hidden:
                    self.hidden_dim = 0
                else:
                    self.hidden_dim = model_weights.get('decoder.hidden_update.weight', torch.zeros(64, 129, 3, 3)).shape[0]
            except:
                model_weights = None
                self.key_dim = config.get('key_dim', 64)
                self.value_dim = config.get('value_dim', 256)
                self.hidden_dim = config.get('hidden_dim', 32)
                self.disable_hidden = (self.hidden_dim <= 0)
        else:
            model_weights = None
            self.key_dim = config.get('key_dim', 64)
            self.value_dim = config.get('value_dim', 256)
            self.hidden_dim = config.get('hidden_dim', 32)
            self.disable_hidden = (self.hidden_dim <= 0)

        config['key_dim'] = self.key_dim
        config['value_dim'] = self.value_dim
        config['hidden_dim'] = self.hidden_dim

        return model_weights

    def load_weights(self, src_dict, init_as_zero_if_needed=False):
        for k in list(src_dict.keys()):
            if k == 'value_encoder.backbone.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    pads = torch.zeros((32, 1, 7, 7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        nn.init.orthogonal_(pads)
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        missing_keys, unexpected_keys = self.load_state_dict(src_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys during weight loading: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys during weight loading: {unexpected_keys}")