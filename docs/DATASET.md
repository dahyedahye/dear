# Datasets

This guide explains how to assemble the `data/` directory that the configs and
scripts expect, and where every piece comes from.

## Target layout

All configs default to `data_root: ./data` with the following structure:

```
data/
├── train/
│   ├── real/coco/                 #  ┐ DEAR-c training (Corvi)
│   ├── real/lsun/                 #  │
│   ├── fake/ldm/                  #  ┘
│   ├── fake/aligned/              #    DEAR-r training (Rajan aligned fakes)
│   ├── fake/lsun_inpaint_sd/      #  ┐ diagnostic inpaint set
│   └── fake/lsun_inpaint_mask/    #  ┘ (inpainted images + masks)
├── val/
│   ├── real/coco/
│   ├── real/lsun/
│   └── fake/ldm/
├── test/
│   ├── real/redcaps/
│   └── fake/{sd,Midjourney,kandinsky,playground,pixelart,lcm,flux,
│             wuerstchen,amused,Chameleon,loki,WildRF}/
└── test_processed/                #   same layout as test/, post-processed
```

The inpaint set uses paired filenames: `{stem}_inpaint.png` in `lsun_inpaint_sd/`
and `{stem}_mask.png` in `lsun_inpaint_mask/`.

## 1. Training data

### DEAR-c: Corvi training set (LSUN/COCO real + LDM fakes)
🤗 **[k-aisi-anti-deepfake/aigi-detection-ldm](https://huggingface.co/datasets/k-aisi-anti-deepfake/aigi-detection-ldm)**

```bash
huggingface-cli download k-aisi-anti-deepfake/aigi-detection-ldm \
    --repo-type dataset --local-dir ./aigi-detection-ldm
# provides data/train/{real/coco, real/lsun, fake/ldm} and data/val/...
```

### DEAR-r: Rajan aligned training set
The aligned real/fake pairs come from **AlignedForensics**:
👉 https://github.com/AniSundar18/AlignedForensics/tree/master/training_code#training-set

Place the aligned fakes under `data/train/fake/aligned/` (paired by filename with
`data/train/real/{coco,lsun}/`).

### Diagnostic inpaint set (DEAR dissection)
🤗 **[k-aisi-anti-deepfake/dear-lsun-inpaint](https://huggingface.co/datasets/k-aisi-anti-deepfake/dear-lsun-inpaint)**

```bash
huggingface-cli download k-aisi-anti-deepfake/dear-lsun-inpaint \
    --repo-type dataset --local-dir ./dear-lsun-inpaint
# the dataset stores data/lsun_inpaint_sd/ and data/lsun_inpaint_mask/.
# place them under data/train/fake/ :
#   data/train/fake/lsun_inpaint_sd/
#   data/train/fake/lsun_inpaint_mask/
```

You can also regenerate this set yourself (see
[`scripts/inpaint_data_gen/commands.sh`](../scripts/inpaint_data_gen/commands.sh)):
`gen_mask.py` samples a random rectangular mask per real LSUN image, then
`gen_inpaint_data.py` inpaints it with Stable Diffusion 1.5.

## 2. Test data (reproduce Table 1)

🤗 **[AniSundar18/LDMFakeDetect](https://huggingface.co/datasets/AniSundar18/LDMFakeDetect)** provides the
real subset (Redcaps) and all 9 generators (SD, Midjourney, Kandinsky,
Playground, PixArt, LCM, FLUX, Wuerstchen, aMUSEd). Place under `data/test/`.

## 3. In-the-wild benchmarks (Section 4.3)

| Benchmark | Link | Notes |
|---|---|---|
| **Chameleon** | https://github.com/shilinyan99/AIDE/issues/7 | place fakes under `data/test/fake/Chameleon/` |
| **LOKI** | https://huggingface.co/datasets/bczhou/LOKI | for the **real** set, reuse Redcaps (the `real` split of [AniSundar18/LDMFakeDetect](https://huggingface.co/datasets/AniSundar18/LDMFakeDetect)) |
| **WildRF** | https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld#wildrf | place under `data/test/fake/WildRF/` |

## 4. Post-processed test set (robustness evaluation)

Build `data/test_processed/` from `data/test/` by applying random JPEG
compression, resizing, and color jitter:

```bash
bash scripts/processed_data_gen/process_test_data.sh
```

> The post-processing protocol follows **AlignedForensics**
> ([issue #9](https://github.com/AniSundar18/AlignedForensics/issues/9)). Credit
> goes to https://github.com/AniSundar18/AlignedForensics.
