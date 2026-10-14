# XMAG: Cross-Magnification Distillation for Efficient Pathology Foundation Model

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-AI4PATH%2FXMAG-yellow)](https://huggingface.co/AI4PATH/XMAG)

<p align="center">
  <img src="./assets/gfigure1.png" width="100%"/>
</p>

<p align="left">
  <img src="./assets/figure2.png" width="45%"/>
</p>

---

**XMAG** is a streamlined **pathology foundation model** trained via **cross-magnification distillation**, enabling accurate and efficient whole-slide image (WSI) encoding at **5× magnification**.  
Despite operating at lower resolution, XMAG achieves performance comparable to state-of-the-art pathology foundation models trained at 20× or 40× magnification — while being up to **30× faster** during inference.

--- 

## Model Checkpoints

Pretrained model weights can be downloaded here:  
👉 [XMAG on Hugging Face](https://huggingface.co/AI4PATH/XMAG)  
👉 [XMAG-5× Checkpoint (Google Drive)](https://drive.google.com/file/d/1RiXa-ZvJ_VorvwgnEffJDIWYTGt_1YBF/view?usp=sharing)

---

## Updates

- 2026-08-20: XMAG is now available on Hugging Face: [AI4PATH/XMAG](https://huggingface.co/AI4PATH/XMAG).

---

## Usage

Install dependencies
```
pip install -r requirements.txt
```

To generate image embeddings, navigate to the `encoding` directory and run:

```bash
python encode.py
```

## Citation
If you use XMAG in your research, please cite:
```
@article{SU2027104352,
title = {Streamline pathology foundation model by cross-magnification distillation},
journal = {Medical Image Analysis},
volume = {115},
pages = {104352},
year = {2027},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2026.104352},
url = {https://www.sciencedirect.com/science/article/pii/S1361841526004214},
author = {Ziyu Su and Abdul Rehman Akbar and Usama Sajjad and Anil V. Parwani and Muhammad Khalid Khan Niazi}}
```


## License
This code is licensed under the **PolyForm Noncommercial License 1.0.0**.  
You may use it for research, academic, or other noncommercial purposes.  
See the [LICENSE](LICENSE) file or visit [polyformproject.org/licenses/noncommercial/1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) for details.
