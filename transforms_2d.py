"""2D crops and augmentations on dict subjects {'image': (C, H, W), 'label': (1, H, W), ...}.

Only augmentations that do not resample the label are implemented (90 degree rotations,
flips and intensity transforms), so the topology of the label is never changed.
"""
import numpy as np
import torch
import torch.nn.functional as F

from transforms_3d import IGNORE_INDEX, foreground_centre_start, hashed_start


class Crop2D:
    """2D counterpart of transforms_3d.OversizedCrop, with the same modes."""

    def __init__(self, target_size, foreground_oversample_percent=0.0, padding_mode="minimum",
                 center_mode=None, align=1):
        self.target_size = tuple(int(s) for s in target_size)
        self.foreground_oversample_percent = float(foreground_oversample_percent)
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
        if self.align > 1:
            start = (start.astype(int) // self.align) * self.align
        start = np.maximum(start.astype(int), 0)

        sl = (slice(None),) + tuple(slice(int(s), int(s) + int(t)) for s, t in zip(start, target))
        return {k: v[sl].clone() if torch.is_tensor(v) and v.dim() == 3 else v
                for k, v in subject.items()}

    @staticmethod
    def _clamped_start(voxel, shape, target):
        start = []
        for i in range(2):
            max_start = max(0, int(shape[i]) - int(target[i]))
            low = max(0, min(int(voxel[i]) - int(target[i]) + 1, max_start))
            high = max(low, min(int(voxel[i]), max_start))
            start.append(np.random.randint(low, high + 1) if high > low else low)
        return np.array(start)

    @staticmethod
    def _sample_uniform(shape, target):
        return np.array([np.random.randint(0, max(1, int(shape[i]) - int(target[i]) + 1))
                         for i in range(2)])

    def _sample_foreground(self, subject, shape, target):
        if "foreground_coords" in subject:
            coords = np.asarray(subject["foreground_coords"])
        else:
            coords = torch.nonzero(subject["label"][0] > 0, as_tuple=False).numpy()
        if len(coords) == 0:
            return self._sample_uniform(shape, target)
        return self._clamped_start(coords[np.random.randint(0, len(coords))], shape, target)

    def _pad(self, subject, shape, target):
        deficit = np.maximum(target - shape, 0)
        lo, hi = deficit // 2, deficit - deficit // 2
        pad = (int(lo[1]), int(hi[1]), int(lo[0]), int(hi[0]))
        out = dict(subject)
        for key, v in subject.items():
            if not torch.is_tensor(v) or v.dim() != 3:
                continue
            if key == "label":
                out[key] = F.pad(v, pad, value=IGNORE_INDEX)
            else:
                fill = {"minimum": float(v.min()), "mean": float(v.float().mean())}.get(self.padding_mode, 0.0)
                out[key] = F.pad(v.float(), pad, value=fill).to(v.dtype)
        if "foreground_coords" in out and np.any(lo > 0) and np.asarray(out["foreground_coords"]).size:
            out["foreground_coords"] = np.asarray(out["foreground_coords"]) + lo[None, :]
        return out


class Rot90:
    def __call__(self, subject):
        h, w = subject["image"].shape[1:]
        if h != w:
            raise ValueError(f"90 degree rotations need a square image, got {h}x{w}")
        k = int(np.random.randint(1, 4))
        out = dict(subject)
        for key in ("image", "label"):
            out[key] = torch.rot90(subject[key], k, dims=(1, 2))
        return out


class Mirror:
    def __init__(self, axes=(0, 1)):
        self.axes = tuple(int(a) for a in axes)

    def __call__(self, subject):
        dims = [a + 1 for a in self.axes if np.random.random() < 0.5]
        out = dict(subject)
        if dims:
            for key in ("image", "label"):
                out[key] = torch.flip(subject[key], dims)
        return out


class _ImageTransform:
    def __call__(self, subject):
        out = dict(subject)
        out["image"] = self.apply(subject["image"])
        return out


class Gamma(_ImageTransform):
    """See transforms_3d.GammaTransform."""

    def __init__(self, gamma_range=(0.7, 1.5), p_invert_image=0.0):
        self.gamma_range = tuple(gamma_range)
        self.p_invert_image = float(p_invert_image)

    def apply(self, x):
        img = x.float().clone()
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
        return img.to(x.dtype)


class Brightness(_ImageTransform):
    def __init__(self, brightness_range=(0.75, 1.25)):
        self.brightness_range = tuple(brightness_range)

    def apply(self, x):
        img = x.float().clone()
        factor = np.random.uniform(*self.brightness_range)
        for c in range(img.shape[0]):
            lo = img[c].min()
            img[c] = (img[c] - lo) * factor + lo
        return img.to(x.dtype)


class Contrast(_ImageTransform):
    def __init__(self, contrast_range=(0.75, 1.25)):
        self.contrast_range = tuple(contrast_range)

    def apply(self, x):
        img = x.float()
        factor = np.random.uniform(*self.contrast_range)
        lo, hi = img.min(), img.max()
        return torch.clamp(img * factor, min=lo, max=hi).to(x.dtype)


class GaussianNoise(_ImageTransform):
    def __init__(self, variance=(0.0, 0.1)):
        self.variance = tuple(variance)

    def apply(self, x):
        std = float(np.sqrt(np.random.uniform(*self.variance)))
        return x if std <= 0 else x + torch.randn_like(x.float()) * std


class GaussianBlur(_ImageTransform):
    """Separable Gaussian blur with reflect padding."""

    def __init__(self, sigma_range=(0.5, 1.0)):
        self.sigma_range = tuple(sigma_range)

    def apply(self, x):
        sigma = float(np.random.uniform(*self.sigma_range))
        if sigma <= 0:
            return x
        r = max(1, int(round(3.0 * sigma)))
        k = torch.exp(-(torch.arange(-r, r + 1, dtype=torch.float32) ** 2) / (2.0 * sigma ** 2))
        k = k / k.sum()
        y = x.float().unsqueeze(0)
        c = y.shape[1]
        y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
        y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
        return y.squeeze(0).to(x.dtype)


class Compose2D:
    """Apply each (transform, probability) pair in order."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, subject):
        for transform, p in self.transforms:
            if p >= 1.0 or np.random.random() < p:
                subject = transform(subject)
        return subject

    def __len__(self):
        return len(self.transforms)


def get_training_transforms(config) -> Compose2D:
    aug = config.augmentation
    resampling = {k: getattr(aug, k) for k in ("rotation_prob", "scaling_prob", "simulate_low_res_prob")
                  if getattr(aug, k) > 0}
    if resampling:
        raise ValueError(f"{resampling}: augmentations that resample the label are not available in 2D")
    t = []
    if aug.rot90_prob > 0:
        t.append((Rot90(), aug.rot90_prob))
    if aug.gaussian_noise_prob > 0:
        t.append((GaussianNoise(aug.gaussian_noise_variance), aug.gaussian_noise_prob))
    if aug.gaussian_blur_prob > 0:
        t.append((GaussianBlur(aug.gaussian_blur_sigma), aug.gaussian_blur_prob))
    if aug.brightness_prob > 0:
        t.append((Brightness(aug.brightness_range), aug.brightness_prob))
    if aug.contrast_prob > 0:
        t.append((Contrast(aug.contrast_range), aug.contrast_prob))
    if aug.gamma_prob > 0:
        t.append((Gamma(aug.gamma_range, p_invert_image=1.0), aug.gamma_prob))
    if aug.gamma_no_invert_prob > 0:
        t.append((Gamma(aug.gamma_range, p_invert_image=0.0), aug.gamma_no_invert_prob))
    if aug.mirror_prob > 0:
        if any(a not in (0, 1) for a in aug.mirror_axes):
            raise ValueError(f"mirror_axes {tuple(aug.mirror_axes)} must be a subset of (0, 1) in 2D")
        t.append((Mirror(aug.mirror_axes), 1.0))
    return Compose2D(t)
