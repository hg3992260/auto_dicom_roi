"""
SammedPredictorLite — 不依赖 albumentations 的 SAM-Med2D 轻量预测器

替代官方 predictor_sammed.py（其依赖 albumentations，在无 C++ 编译器时无法安装）。
预处理逻辑等价：长边缩放到模型 img_size + 像素归一化，只是用 cv2/torch 手动实现。
"""
import numpy as np
import torch
from torch.nn import functional as F
import cv2


class SammedPredictorLite:
    def __init__(self, sam_model):
        self.model = sam_model
        self.device = sam_model.device
        self.reset_image()

    def reset_image(self):
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.new_size = None

    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        assert image_format in ("RGB", "BGR"), \
            f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # 像素归一化
        pixel_mean = self.model.pixel_mean.squeeze().cpu().numpy()
        pixel_std = self.model.pixel_std.squeeze().cpu().numpy()
        input_image = (image.astype(np.float32) - pixel_mean) / pixel_std

        ori_h, ori_w, _ = input_image.shape
        self.original_size = (ori_h, ori_w)
        new_size = self.model.image_encoder.img_size
        self.new_size = (new_size, new_size)

        # 长边缩放（cv2.resize 保持纵横比）
        img_float = cv2.resize(input_image, (new_size, new_size),
                               interpolation=cv2.INTER_NEAREST)
        # HWC -> CHW -> BCHW
        tensor = torch.as_tensor(img_float, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            self.features = self.model.image_encoder(tensor)
        self.is_image_set = True

    def get_image_embedding(self) -> torch.Tensor:
        assert self.is_image_set, "image must be set first"
        return self.features

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
    ):
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        coords_torch, labels_torch, box_torch = None, None, None
        if point_coords is not None:
            point_coords = self._apply_coords(point_coords, self.original_size, self.new_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch = coords_torch[None, :, :]
            labels_torch = labels_torch[None, :]
        if box is not None:
            box = self._apply_boxes(box, self.original_size, self.new_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]

        masks, iou_pred, low_res = self.predict_torch(
            coords_torch, labels_torch, box_torch,
            multimask_output=multimask_output, return_logits=return_logits)

        masks = masks[0].detach().cpu().numpy()
        iou = iou_pred[0].detach().cpu().numpy()
        return masks, iou, None

    @torch.no_grad()
    def predict_torch(self, point_coords=None, point_labels=None,
                      boxes=None, multimask_output=True, return_logits=False):
        if not self.is_image_set:
            raise RuntimeError("image must be set first")

        # 稀疏提示嵌入
        sparse_embeddings = None
        dense_embeddings = None
        if point_coords is not None:
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=(point_coords, point_labels), boxes=None, masks=None)
        elif boxes is not None:
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None, boxes=boxes, masks=None)
        else:
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None, boxes=None, masks=None)

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # 后处理：从低分辨率上采样到 new_size
        masks = F.interpolate(
            low_res_masks,
            size=self.new_size,
            mode="bilinear",
            align_corners=False,
        )

        # 裁剪到原始尺寸
        masks = masks[..., :self.original_size[0], :self.original_size[1]]
        return masks, iou_predictions, low_res_masks

    def _apply_coords(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = np.array(coords, dtype=float).copy()
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def _apply_boxes(self, boxes, original_size, new_size):
        boxes = np.array(boxes, dtype=float)
        # xyxy -> xywh -> xyxy
        boxes = boxes.copy()
        boxes[..., 2] = boxes[..., 2] - boxes[..., 0]  # w
        boxes[..., 3] = boxes[..., 3] - boxes[..., 1]  # h
        boxes = self._apply_coords(boxes, original_size, new_size)
        boxes[..., 2] = boxes[..., 2] + boxes[..., 0]  # x2
        boxes[..., 3] = boxes[..., 3] + boxes[..., 1]  # y2
        return boxes

    def postprocess_masks(self, masks, new_size, original_size):
        masks = masks[..., :new_size[0], :new_size[1]]
        masks = F.interpolate(
            masks, size=(original_size[0], original_size[1]),
            mode="bilinear", align_corners=False)
        return masks
