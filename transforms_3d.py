"""3D crops and augmentations on TorchIO subjects, following the nnU-Net augmentation scheme."""
import hashlib
from math import sqrt

import numpy as np
import torch
import torchio as tio

IGNORE_INDEX = -1


class RotationScalingTransform:
    """Rotation and scaling with independent probabilities, in one interpolation."""

    def __init__(self, rotation_range, scale_range, rotation_p=0.2, scaling_p=0.2):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.rotation_p = rotation_p
        self.scaling_p = scaling_p

    def __call__(self, subject):
        rotate = np.random.random() < self.rotation_p
        scale = np.random.random() < self.scaling_p
        if not rotate and not scale:
            return subject
        return tio.RandomAffine(
            scales=self.scale_range if scale else (1.0, 1.0),
            degrees=tuple(self.rotation_range) * 3 if rotate else (0, 0, 0, 0, 0, 0),
            translation=0, isotropic=True, image_interpolation="linear",
            label_interpolation="nearest", default_pad_value="mean",
            default_pad_label=IGNORE_INDEX, p=1.0, copy=False)(subject)


class Rot90Transform:
    """Rotation by 90, 180 or 270 degrees in the plane of two patch axes. Unlike an
    arbitrary rotation it does not resample, so the topology of the label is unchanged."""

    def __init__(self, p=0.75, axes=(0, 1)):
        self.p = p
        self.axes = tuple(axes)

    def __call__(self, subject):
        if np.random.random() >= self.p:
            return subject
        dims = (self.axes[0] + 1, self.axes[1] + 1)
        images = subject.get_images_dict(intensity_only=False)
        shape = next(iter(images.values())).data.shape
        k = int(np.random.randint(1, 4)) if shape[dims[0]] == shape[dims[1]] else 2
        for image in images.values():
            image.set_data(torch.rot90(image.data, k, dims=dims).contiguous())
        return subject


class GammaTransform:
    """Gamma on the intensity range, optionally on the inverted image, keeping mean and
    standard deviation (batchgenerators' gamma transform)."""

    def __init__(self, gamma_range=(0.7, 1.5), p_invert_image=1.0):
        self.gamma_range = gamma_range
        self.p_invert_image = p_invert_image

    def __call__(self, img):
        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        invert = np.random.random() < self.p_invert_image
        for c in range(img.shape[0]):
            if invert:
                img[c] *= -1
            mean, std = torch.mean(img[c]), torch.std(img[c])
            minm = torch.min(img[c])
            rnge = torch.max(img[c]) - minm
            img[c] = torch.pow((img[c] - minm) / torch.clamp(rnge, min=1e-7), gamma) * rnge + minm
            mean_now, std_now = torch.mean(img[c]), torch.std(img[c])
            img[c] -= mean_now
            img[c] *= std / torch.clamp(std_now, min=1e-7)
            img[c] += mean
            if invert:
                img[c] *= -1
        return img


class ContrastTransform:
    """Multiply by a factor and clip to the original range."""

    def __init__(self, contrast_range=(0.75, 1.25)):
        self.contrast_range = contrast_range

    def __call__(self, img):
        factor = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
        lo, hi = img.min(), img.max()
        return torch.clamp(img * factor, min=lo, max=hi)


class BrightnessTransform:
    """Multiply each channel by a factor, relative to its minimum."""

    def __init__(self, brightness_range=(0.75, 1.25)):
        self.brightness_range = brightness_range

    def __call__(self, img):
        factor = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        for c in range(img.shape[0]):
            lo = img[c].min()
            img[c] = (img[c] - lo) * factor + lo
        return img


class OversizedCrop:
    """Crop a subject to `target_size`.

    center_mode None samples the origin (with probability `foreground_oversample_percent`
    around a random foreground voxel, otherwise uniformly). The deterministic modes are
    'volume' (array centre), 'foreground' (centre of the label bounding box) and
    'random_fixed' (an origin hashed from case id and patch index, fixed across epochs).
    Volumes smaller than the target are padded; the label is padded with IGNORE_INDEX.
    """

    def __init__(self, target_size, foreground_oversample_percent=0.0, padding_mode="minimum",
                 center_mode=None, align=1):
        self.target_size = tuple(int(s) for s in target_size)
        self.foreground_oversample_percent = foreground_oversample_percent
        self.padding_mode = padding_mode
        self.center_mode = center_mode
        self.align = max(1, int(align))

    def __call__(self, subject):
        shape = np.array(subject["image"].shape[1:])
        target = np.array(self.target_size)
        if np.any(shape < target):
            subject = self._pad(subject, shape, target)
            shape = np.array(subject["image"].shape[1:])

        sample_fg = (self.center_mode is None and self.foreground_oversample_percent > 0
                     and np.random.random() < self.foreground_oversample_percent)
        if self.center_mode == "random_fixed":
            start = hashed_start(subject, shape, target)
        elif self.center_mode == "foreground":
            start = foreground_centre_start(subject, shape, target)
        elif sample_fg:
            start = self._sample_foreground(subject, shape, target)
        elif self.center_mode == "volume":
            start = (shape - target) // 2
        else:
            start = self._sample_uniform(shape, target)
        start = [max(0, (int(s) // self.align) * self.align) for s in start]

        sl = tuple(slice(s, s + t) for s, t in zip(start, target))
        out = {"case_id": subject["case_id"]}
        for key in ("image", "label"):
            cls = tio.ScalarImage if key == "image" else tio.LabelMap
            out[key] = cls(tensor=subject[key][sl].data.clone(), affine=subject[key].affine)
        if "foreground_coords" in subject:
            out["foreground_coords"] = subject["foreground_coords"]
        return tio.Subject(**out)

    @staticmethod
    def _sample_uniform(shape, target):
        return np.array([np.random.randint(0, max(1, int(shape[i]) - int(target[i]) + 1))
                         for i in range(3)])

    def _sample_foreground(self, subject, shape, target):
        if "foreground_coords" in subject:
            coords = subject["foreground_coords"]
        else:
            coords = torch.nonzero(subject["label"].data[0] > 0, as_tuple=False).numpy()
        if len(coords) == 0:
            return self._sample_uniform(shape, target)
        voxel = coords[np.random.randint(0, len(coords))]
        start = []
        for i in range(3):
            max_start = max(0, int(shape[i]) - int(target[i]))
            low = max(0, min(int(voxel[i]) - int(target[i]) + 1, max_start))
            high = max(low, min(int(voxel[i]), max_start))
            start.append(np.random.randint(low, high + 1) if high > low else low)
        return np.array(start)

    def _pad(self, subject, shape, target):
        deficit = np.maximum(target - shape, 0)
        lo, hi = deficit // 2, deficit - deficit // 2
        padded = tio.Pad(tuple(int(x) for pair in zip(lo, hi) for x in pair),
                         padding_mode=self.padding_mode)(subject)
        for value in padded.values():
            if isinstance(value, tio.LabelMap):
                data = value.data
                for axis in range(3):
                    idx = [slice(None)] * 4
                    if lo[axis]:
                        idx[axis + 1] = slice(0, lo[axis])
                        data[tuple(idx)] = IGNORE_INDEX
                    if hi[axis]:
                        idx[axis + 1] = slice(data.shape[axis + 1] - hi[axis], None)
                        data[tuple(idx)] = IGNORE_INDEX
        if "foreground_coords" in padded and np.any(lo > 0) and np.asarray(padded["foreground_coords"]).size:
            padded["foreground_coords"] = np.asarray(padded["foreground_coords"]) + lo[None, :]
        return padded


def hashed_start(subject, shape, target):
    """Crop origin derived from sha256(case id, patch index, axis)."""
    case_id = subject.get("case_id", "")
    patch_index = int(subject.get("patch_index", 0) or 0)
    start = []
    for axis in range(len(shape)):
        span = int(shape[axis]) - int(target[axis])
        if span <= 0:
            start.append(0)
            continue
        draw = int.from_bytes(hashlib.sha256(f"{case_id}|{patch_index}|{axis}".encode()).digest()[:8], "big")
        start.append(draw % (span + 1))
    return np.array(start)


def foreground_centre_start(subject, shape, target):
    """Crop origin that centres the window on the bounding box of the label."""
    if "foreground_coords" in subject:
        coords = np.asarray(subject["foreground_coords"])
    else:
        coords = np.argwhere(np.asarray(subject["label"].data[0] if hasattr(subject["label"], "data")
                                        else subject["label"][0]) > 0)
    centre = shape / 2.0 if coords.size == 0 else (coords.min(axis=0) + coords.max(axis=0)) / 2.0
    start = np.round(centre - np.asarray(target) / 2.0).astype(int)
    return np.clip(start, 0, np.maximum(np.asarray(shape) - np.asarray(target), 0))


def get_training_transforms(config) -> tio.Compose:
    aug = config.augmentation
    t = []
    if aug.rotation_prob > 0 or aug.scaling_prob > 0:
        t.append(RotationScalingTransform(aug.rotation_range, aug.scale_range,
                                          aug.rotation_prob, aug.scaling_prob))
    if aug.rot90_prob > 0:
        t.append(Rot90Transform(aug.rot90_prob, aug.rot90_axes))
    if aug.gaussian_noise_prob > 0:
        t.append(tio.RandomNoise(mean=0, std=tuple(sqrt(v) for v in aug.gaussian_noise_variance),
                                 p=aug.gaussian_noise_prob, include=["image"], copy=False))
    if aug.gaussian_blur_prob > 0:
        t.append(tio.RandomBlur(std=aug.gaussian_blur_sigma, p=aug.gaussian_blur_prob,
                                include=["image"], copy=False))
    if aug.brightness_prob > 0:
        t.append(tio.Lambda(BrightnessTransform(aug.brightness_range), p=aug.brightness_prob,
                            include=["image"], copy=False))
    if aug.contrast_prob > 0:
        t.append(tio.Lambda(ContrastTransform(aug.contrast_range), p=aug.contrast_prob,
                            include=["image"], copy=False))
    if aug.simulate_low_res_prob > 0:
        lo, hi = aug.low_res_scale_range
        t.append(tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(max(1.0, 1.0 / hi), 1.0 / lo),
                                      image_interpolation="linear", p=aug.simulate_low_res_prob,
                                      include=["image"], copy=False))
    if aug.gamma_prob > 0:
        t.append(tio.Lambda(GammaTransform(aug.gamma_range, p_invert_image=1.0), p=aug.gamma_prob,
                            include=["image"], copy=False))
    if aug.gamma_no_invert_prob > 0:
        t.append(tio.Lambda(GammaTransform(aug.gamma_range, p_invert_image=0.0),
                            p=aug.gamma_no_invert_prob, include=["image"], copy=False))
    if aug.mirror_prob > 0:
        t.append(tio.RandomFlip(axes=aug.mirror_axes, flip_probability=aug.mirror_prob, copy=False))
    return tio.Compose(t)
