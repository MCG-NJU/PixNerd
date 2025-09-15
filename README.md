# PixNerd: Pixel Neural Field Diffusion
<div style="text-align: center;">
  <a href="http://arxiv.org/abs/2507.23268"><img src="https://img.shields.io/badge/arXiv-2507.23268-b31b1b.svg" alt="arXiv"></a>
    <a href="https://huggingface.co/spaces/MCG-NJU/PixNerd"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Online_Demo-green" alt="arXiv"></a>  
</div>

![](./figs/arch.png)

## Introduction
We propose PixNerd, a powerful and efficient **pixel-space** diffusion transformer for image generation (without VAE). Different from conventional pixel diffusion models, we employ the neural field to improve the high frequercy modeling .

* We achieve **1.93 FID** on ImageNet256x256 Benchmark with PixNerd-XL/16 (1600k training steps).
* We achieve **2.84 FID** on ImageNet512x512 Benchmark with PixNerd-XL/16.
* We achieve **0.73 overall score** on GenEval Benchmark with PixNerd-XXL/16.
* We achieve **80.9 avergae score** on DPG Benchmark with PixNerd-XXL/16.

## Visualizations
![](./figs/pixelnerd_teaser.png)
![](./figs/pixnerd_multires.png)
## Checkpoints

| Dataset       | Model         | Params | FID   | HuggingFace                           |
|---------------|---------------|--------|-------|---------------------------------------|
| ImageNet256   | PixNerd-XL/16 | 700M   | 1.93  | [🤗](https://huggingface.co/MCG-NJU/PixNerd-XL-P16-C2I) |
| ImageNet512   | PixNerd-XL/16 | 700M   | 2.84  | [🤗](https://huggingface.co/MCG-NJU/PixNerd-XL-P16-C2I) |

| Dataset       | Model         | Params | GenEval | DPG  | HuggingFace                                              |
|---------------|---------------|--------|------|------|----------------------------------------------------------|
| Text-to-Image | PixNerd-XXL/16| 1.2B | 0.73 | 80.9 | [🤗](https://huggingface.co/MCG-NJU/PixNerd-XXL-P16-T2I) |
## Online Demos
![](./figs/demo.png)
We provide online demos for PixNerd-XXL/16(text-to-image) on HuggingFace Spaces.

强烈建议本地部署玩玩，线上的模型推理速度会慢一些。以及因为这个我把任意分辨率和动画都关了。

HF spaces: [https://huggingface.co/spaces/MCG-NJU/PixNerd](https://huggingface.co/spaces/MCG-NJU/PixNerd)

To host the local gradio demo, run the following command:
```bash
# for text-to-image applications
python app.py --config configs_t2i/inference_heavydecoder.yaml  --ckpt_path=XXX.ckpt
```

## Usages
For C2i(ImageNet), We use ADM evaluation suite to report FID.
```bash
# for installation
pip install -r requirements.txt
```

```bash
# for inference
python main.py predict -c configs_c2i/pix256std1_repa_pixnerd_xl.yaml --ckpt_path=XXX.ckpt
# # or specify the GPU(s) to use with as :
CUDA_VISIBLE_DEVICES=0,1, python main.py predict -c configs_c2i/pix256std1_repa_pixnerd_xl.yaml --ckpt_path=XXX.ckpt
```

```bash
# for training
# train
python main.py fit -c configs_c2i/pix256std1_repa_pixnerd_xl.yaml
```
For T2i, we use GenEval and DPG to collect metrics.

## Reference
```bibtex
@article{2507.23268,
Author = {Shuai Wang and Ziteng Gao and Chenhui Zhu and Weilin Huang and Limin Wang},
Title = {PixNerd: Pixel Neural Field Diffusion},
Year = {2025},
Eprint = {arXiv:2507.23268},
}
```

## Acknowledgement
The code is mainly built upon [FlowDCN](https://github.com/MCG-NJU/FlowDCN) and [DDT](https://github.com/MCG-NJU/DDT).
