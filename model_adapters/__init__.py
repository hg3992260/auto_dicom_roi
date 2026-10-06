"""
model_adapters — 多模型 SAM 家族统一加载与推理分派层

支持模型：
  - sam           : 官方 SAM ViT-B/L/H (segment_anything)
  - medsam        : 官方 MedSAM ViT-B (兼容 segment_anything registry)
  - litemedsam    : LiteMedSAM (TinyViT 结构)
  - sam2          : SAM2.1 Hiera-Tiny (sam2 包)
  - sam_med2d     : SAM-Med2D (adapter 结构)
  - sam_med3d     : SAM-Med3D (3D 结构)

每个模型族提供统一的 SamLike 接口：
  - load_model()                    -> 返回模型实例
  - set_image(image_rgb)            -> 设置图像
  - predict(points, labels)         -> 点提示分割
  - predict_boxes(boxes)            -> 框提示分割
  - auto_segment(image_rgb)         -> 自动分割
"""

import os
import sys
import logging
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

# Windows: torch + onnxruntime 并存时 OpenMP 重复加载，需允许
if os.name == 'nt':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
# 模型族识别
# ═══════════════════════════════════════════════

FAMILY_HINTS = [
    # (family, substrings in filename)
    ('sam2', ['sam2.1', 'sam2.1_hiera', 'sam2_hiera']),
    ('litemedsam', ['lite_medsam', 'litemedsam']),
    ('medsam', ['medsam']),
    ('sam_med3d', ['sam_med3d', 'med3d']),
    ('sam_med2d', ['sam_med2d', 'med2d']),
    ('sam', ['sam_vit']),
]

FAMILY_BY_KEY = {
    'vit_b': 'sam', 'vit_l': 'sam', 'vit_h': 'sam',
}


def detect_family(path: str) -> str:
    """根据权重文件路径识别模型族"""
    fname = os.path.basename(str(path)).lower()
    for family, hints in FAMILY_HINTS:
        for h in hints:
            if h in fname:
                return family
    return 'sam'


def describe_family(family: str) -> str:
    desc = {
        'sam': 'SAM (Meta official)',
        'medsam': 'MedSAM (medical)',
        'litemedsam': 'LiteMedSAM (lightweight medical)',
        'sam2': 'SAM2 (Hiera)',
        'sam_med2d': 'SAM-Med2D',
        'sam_med3d': 'SAM-Med3D',
    }
    return desc.get(family, family)


# ═══════════════════════════════════════════════
# 各模型族加载器
# ═══════════════════════════════════════════════

def _import_segment_anything():
    """优先使用本地 MedSAM fork，回退到 pip segment_anything"""
    local = os.path.join(os.path.dirname(__file__), 'medsam', 'segment_anything')
    if os.path.isdir(local) and local not in sys.path:
        sys.path.insert(0, os.path.dirname(local))
    try:
        from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
        return sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError("segment_anything not available")


def _import_litemedsam():
    """加载 LiteMedSAM 所需模块 (TinyViT + MedSAM_Lite)"""
    local = os.path.join(os.path.dirname(__file__), 'medsam', 'segment_anything')
    # 弹出缓存的 segment_anything，强制使用 medsam fork
    for name in list(sys.modules):
        if name == 'segment_anything' or name.startswith('segment_anything.'):
            del sys.modules[name]
    if os.path.isdir(local) and local not in sys.path:
        sys.path.insert(0, os.path.dirname(local))
    try:
        from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
        from segment_anything.modeling.tiny_vit_sam import TinyViT
        return MaskDecoder, PromptEncoder, TwoWayTransformer, TinyViT
    except ImportError as e:
        raise ImportError("LiteMedSAM modules not available: %s" % e)


def _import_sam2():
    """加载 sam2 包"""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return build_sam2, SAM2ImagePredictor
    except ImportError as e:
        raise ImportError("sam2 package not available: %s" % e)


# ═══════════════════════════════════════════════
# 模型加载分派
# ═══════════════════════════════════════════════

def load_model(path: str, device: str = "cpu", family: Optional[str] = None):
    """
    根据模型族加载对应模型，返回 (model, family)

    返回的 model 为各家族原生实例，供对应 predictor 使用。
    """
    if family is None:
        family = detect_family(path)

    logger.info("Loading model family=%s path=%s device=%s", family, path, device)

    if family == 'sam':
        sam_model_registry, _, _ = _import_segment_anything()
        model_type = _detect_sam_type(path)
        model = sam_model_registry[model_type](checkpoint=path)
        model.to(device=device)
        return model, family

    if family == 'medsam':
        # Official MedSAM uses standard SAM keys — load via segment_anything
        sam_model_registry, _, _ = _import_segment_anything()
        model = sam_model_registry['vit_b'](checkpoint=path)
        model.to(device=device)
        return model, family

    if family == 'litemedsam':
        return _load_litemedsam(path, device), family

    if family == 'sam2':
        return _load_sam2(path, device), family

    if family == 'sam_med2d':
        return _load_sam_med2d(path, device), family

    if family == 'sam_med3d':
        return _load_sam_med3d(path, device), family

    raise ValueError("Unknown model family: %s" % family)


def _detect_sam_type(path: str) -> str:
    """从文件名推断 SAM 类型"""
    fname = os.path.basename(str(path)).lower()
    if 'vit_h' in fname or 'vit-h' in fname:
        return 'vit_h'
    if 'vit_l' in fname or 'vit-l' in fname:
        return 'vit_l'
    return 'vit_b'


def _load_litemedsam(path: str, device: str = "cpu"):
    """加载 LiteMedSAM (TinyViT encoder)"""
    import torch
    import torch.nn as nn
    from functools import partial

    MaskDecoder, PromptEncoder, TwoWayTransformer, TinyViT = _import_litemedsam()

    # 组装 LiteMedSAM
    image_encoder = TinyViT(
        img_size=256, in_chans=3,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4., drop_rate=0., drop_path_rate=0.0,
        use_checkpoint=False, mbconv_expand_ratio=4.0,
        local_conv_size=3, layer_lr_decay=0.8,
    )
    prompt_encoder = PromptEncoder(
        embed_dim=256, image_embedding_size=(64, 64),
        input_image_size=(256, 256), mask_in_chans=16,
    )
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8,
        ),
        transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256,
    )

    class MedSAM_Lite(nn.Module):
        def __init__(self, image_encoder, mask_decoder, prompt_encoder):
            super().__init__()
            self.image_encoder = image_encoder
            self.mask_decoder = mask_decoder
            self.prompt_encoder = prompt_encoder

        def forward(self, image, boxes=None):
            image_embedding = self.image_encoder(image)
            with torch.no_grad():
                if boxes is None:
                    sparse, dense = self.prompt_encoder(
                        points=None, boxes=None, masks=None)
                else:
                    if isinstance(boxes, np.ndarray):
                        boxes = torch.as_tensor(boxes, dtype=torch.float32,
                                                device=image.device)
                    if boxes.dim() == 2:
                        boxes = boxes[:, None, :]  # (B, 1, 4)
                    sparse, dense = self.prompt_encoder(
                        points=None, boxes=boxes, masks=None)
            low_res_masks, iou = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            return low_res_masks

    model = MedSAM_Lite(image_encoder, mask_decoder, prompt_encoder)
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)
    model.to(device=device)
    model.eval()
    return model


def _load_sam2(path: str, device: str = "cpu"):
    """加载 SAM2 模型"""
    build_sam2, _ = _import_sam2()
    # 根据权重文件名选择配置
    fname = os.path.basename(str(path)).lower()
    if 'tiny' in fname:
        cfg = 'sam2.1_hiera_t.yaml'
    elif 'large' in fname:
        cfg = 'sam2.1_hiera_l.yaml'
    elif 'small' in fname:
        cfg = 'sam2.1_hiera_s.yaml'
    else:
        cfg = 'sam2.1_hiera_t.yaml'
    model = build_sam2(cfg, path, device=device)
    return model


def _load_sam_med2d(path: str, device: str = "cpu"):
    """加载 SAM-Med2D（含 encoder adapter）"""
    import torch
    from functools import partial

    # 用唯一包名 sa_med2d_pkg 导入（避免与 segment_anything 冲突）
    pkg_parent = os.path.join(os.path.dirname(__file__), 'sam_med2d')
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)

    import importlib
    try:
        pkg = importlib.import_module('sa_med2d_pkg')
        from sa_med2d_pkg.modeling import (
            Sam, ImageEncoderViT, MaskDecoder, PromptEncoder, TwoWayTransformer)
    except Exception as e:
        raise RuntimeError('SAM-Med2D module load failed: %s' % e)

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model', ckpt)

    # SAM-Med2D 权重基于 256x256 训练
    image_size = 256
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    model = Sam(
        image_encoder=ImageEncoderViT(
            depth=12, embed_dim=768, img_size=image_size, mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12, patch_size=vit_patch_size, qkv_bias=True,
            use_rel_pos=True, global_attn_indexes=[2, 5, 8, 11],
            window_size=14, out_chans=prompt_embed_dim,
            adapter_train=True,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim,
                mlp_dim=2048, num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3, iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print('[SAM-Med2D] missing keys: %d' % len(missing))
        if unexpected:
            print('[SAM-Med2D] unexpected keys: %d' % len(unexpected))
    except Exception as e:
        raise RuntimeError('SAM-Med2D load failed: %s' % e)

    model.to(device=device)
    model.eval()
    return model


def _load_sam_med3d(path: str, device: str = "cpu"):
    """加载 SAM-Med3D（3D 结构，输入为体数据）"""
    import torch
    from functools import partial

    # 用唯一包名 sa_med3d_pkg 导入
    pkg_parent = os.path.join(os.path.dirname(__file__), 'sam_med3d')
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)

    import importlib
    try:
        pkg = importlib.import_module('sa_med3d_pkg')
        from sa_med3d_pkg.build_sam3D import build_sam3D_vit_b, build_sam3D_vit_b_ori
        # turbo 权重为 100.5M，匹配 vit_b_ori (768 维)
        try:
            model = build_sam3D_vit_b_ori(checkpoint=path)
        except Exception as e1:
            print('[SAM-Med3D] vit_b_ori failed: %s' % str(e1)[:100])
            model = build_sam3D_vit_b(checkpoint=path)
    except Exception as e:
        raise RuntimeError('SAM-Med3D module load failed: %s' % e)

    model.to(device=device)
    model.eval()
    return model


# ═══════════════════════════════════════════════
# 统一 Predictor 接口
# ═══════════════════════════════════════════════

class UnifiedPredictor:
    """统一推理接口，兼容各模型族"""

    def __init__(self, model, family: str, device: str = "cpu"):
        self.model = model
        self.family = family
        self.device = device
        self._predictor = None
        self._mask_generator = None
        self._current_image = None
        self._image_rgb = None
        self._build_predictor()

    def _build_predictor(self):
        """根据模型族构建 predictor"""
        if self.family in ('sam', 'medsam'):
            from segment_anything import SamPredictor, SamAutomaticMaskGenerator
            self._predictor = SamPredictor(self.model)
            self._mask_generator = SamAutomaticMaskGenerator(
                model=self.model, points_per_side=32,
                pred_iou_thresh=0.86, stability_score_thresh=0.92,
                crop_n_layers=1, crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
        elif self.family == 'litemedsam':
            # LiteMedSAM 用框提示，需自定义
            self._build_litemedsam_predictor()
        elif self.family == 'sam2':
            build_sam2, SAM2ImagePredictor = _import_sam2()
            self._predictor = SAM2ImagePredictor(self.model)
        elif self.family == 'sam_med2d':
            self._build_med2d_predictor()
        elif self.family == 'sam_med3d':
            # SAM-Med3D 需要 3D 体数据输入，标注为受限
            self._predictor = None
            self._med3d = True
        else:
            raise NotImplementedError("Predictor for %s not implemented" % self.family)

    def _build_med2d_predictor(self):
        """SAM-Med2D predictor：优先官方 SammedPredictor，缺 albumentations 时用轻量版"""
        try:
            from sa_med2d_pkg.predictor_sammed import SammedPredictor
            self._predictor = SammedPredictor(self.model)
        except ImportError:
            # albumentations 不可用时使用轻量实现
            lite_mod = os.path.join(os.path.dirname(__file__), 'sam_med2d', 'sammed_predictor_lite.py')
            import importlib.util
            spec = importlib.util.spec_from_file_location('sammed_predictor_lite', lite_mod)
            mod = importlib.util.module_from_spec(spec)
            sys.modules['sammed_predictor_lite'] = mod
            spec.loader.exec_module(mod)
            self._predictor = mod.SammedPredictorLite(self.model)

    def _build_med2d_predictor_fallback(self):
        """备用：无 adapter predictor 时的 2D 点提示"""
        import numpy as np
        import torch

        def _predict_pts(img_rgb, points, labels):
            # 用 model 直接前向（需按 SAM-Med2D 预处理）
            raise NotImplementedError(
                "SAM-Med2D fallback predictor requires preprocessing pipeline")

        self._med2d_fallback = _predict_pts

    def _build_litemedsam_predictor(self):
        """LiteMedSAM 的简单框提示预测"""
        import numpy as np
        import cv2
        import torch

        def _predict_box(img_rgb, boxes_np):
            # 预处理到 256
            h, w = img_rgb.shape[:2]
            scale = 256.0 / max(h, w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            img_256 = cv2.resize(img_rgb, (nw, nh))
            pad_h, pad_w = 256 - nh, 256 - nw
            img_padded = np.pad(img_256, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            # 归一化
            img_norm = (img_padded - img_padded.min()) / np.clip(
                img_padded.max() - img_padded.min(), 1e-8, None)
            tensor = torch.as_tensor(img_norm, dtype=torch.float32, device=self.device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            # 框缩放
            boxes_scaled = np.array(boxes_np, dtype=float) * scale
            out = self.model(tensor, boxes=boxes_scaled)
            # 后处理
            mask = (out[0, 0] > 0.5).cpu().numpy()
            mask = mask[:nh, :nw]
            mask = cv2.resize(mask.astype(np.uint8), (w, h)).astype(bool)
            return mask

        self._litemedsam_predict = _predict_box

    def set_image(self, image_rgb):
        import numpy as np
        self._image_rgb = image_rgb
        if self.family in ('sam', 'medsam', 'sam2', 'sam_med2d'):
            self._predictor.set_image(image_rgb)
        self._current_image = image_rgb

    def predict(self, points=None, labels=None, **kwargs):
        """点提示分割，返回 (masks, scores, logits)

        兼容两种调用风格：
          predict(points, labels)
          predict(point_coords=..., point_labels=..., multimask_output=...)
        """
        import numpy as np
        point_coords = kwargs.get('point_coords', points)
        point_labels = kwargs.get('point_labels', labels)
        multimask = kwargs.get('multimask_output', True)
        box = kwargs.get('box', None)
        return_logits = kwargs.get('return_logits', False)

        if self.family in ('sam', 'medsam'):
            masks, scores, logits = self._predictor.predict(
                point_coords=point_coords, point_labels=point_labels,
                box=box, multimask_output=multimask)
            return masks, scores, logits
        elif self.family == 'sam2':
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords, point_labels=point_labels,
                box=box, multimask_output=multimask,
                return_logits=True)
            return masks, scores, None
        elif self.family == 'sam_med2d':
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords, point_labels=point_labels,
                multimask_output=multimask, return_logits=True)
            return masks, scores, None
        elif self.family == 'litemedsam':
            # LiteMedSAM 官方仅支持框提示；点提示转为中心小框
            if point_coords is not None and len(point_coords) > 0:
                pts = np.asarray(point_coords, dtype=float)
                if pts.ndim == 1:
                    pts = pts[None, :]
                # 以点为中心构造 60x60 框（原图坐标）
                boxes = []
                for p in pts:
                    x, y = float(p[0]), float(p[1])
                    boxes.append([x - 30, y - 30, x + 30, y + 30])
                mask = self._litemedsam_predict(self._image_rgb, boxes)
                # 生成 (N, H, W) 掩膜 + 分数
                masks = mask[None].astype(np.uint8)
                scores = np.array([1.0])
                return masks, scores, None
            raise RuntimeError("LiteMedSAM requires point prompts")
        else:
            raise NotImplementedError("Point predict for %s not implemented" % self.family)

    def predict_boxes(self, boxes) -> np.ndarray:
        """框提示分割，返回二值掩膜"""
        import numpy as np
        if self.family == 'litemedsam':
            return self._litemedsam_predict(self._image_rgb, boxes)
        if self.family == 'sam2':
            masks, _, _ = self._predictor.predict(
                point_coords=None, point_labels=None,
                box=boxes, multimask_output=False)
            return masks[0]
        if self.family in ('sam', 'medsam'):
            masks, _, _ = self._predictor.predict(
                point_coords=None, point_labels=None,
                box=np.array(boxes)[None, :], multimask_output=False)
            return masks[0]
        raise NotImplementedError("Box predict for %s not implemented" % self.family)

    def auto_segment(self, image_rgb):
        """自动分割，返回 SAM mask dict 列表"""
        if self.family in ('sam', 'medsam'):
            return self._mask_generator.generate(image_rgb)
        # 其他模型族：尝试用框/点网格近似，或标记不支持
        raise NotImplementedError("Auto-segment for %s not implemented" % self.family)

    def generate(self, image_rgb):
        """SamAutomaticMaskGenerator 兼容接口"""
        return self.auto_segment(image_rgb)


def create_predictor(path: str, device: str = "cpu",
                     family: Optional[str] = None) -> UnifiedPredictor:
    """便捷入口：加载模型并创建统一 predictor"""
    model, family = load_model(path, device, family)
    return UnifiedPredictor(model, family, device)
