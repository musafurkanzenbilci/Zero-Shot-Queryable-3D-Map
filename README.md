<h1 align="center">Zero-shot Open-Vocabulary Language Queryable 3D Semantic Map</h1>

__Inspired by the [**ConceptFusion**](https://arxiv.org/abs/2302.07241) and [**CLIP-Fields**](https://arxiv.org/abs/2210.05663) papers, a Python implementation of a zero shot open-vocabulary language queryable 3D semantic map creation from a video system.__

<!-- # th table of contents -->
<p align="center">
  <!-- <video src="keyboard_clean.mov" controls width="480">
    Your browser does not support the video tag. <br>
    <a href="keyboard_clean.mov">Download the demo video</a>
  </video> -->

  <img src="img/keyboard_clean.gif" controls width="640"></img>
</p>


<!-- # DIAGRAM IMAGE  -->

# Running the demo
## Environment setup
```python
pip install -r requirements.txt
```
## Dataset
Download the sequence __freiburg3_long_office_household__ from [TUM Dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) and specify the extracted location in the pipeline command

## Run the pipeline
```python
python3 pipeline.py --interactive --data <DATASET_ROOT>
```

<!-- ## Demo Video -->

<!-- <p align="center">
  <video src="demo.mov" controls width="640">
    <br>
    <a href="demo.mov">Download the demo video</a>
  </video>
</p> -->

<!-- 
# Dataset

# Usage -->

# Roadmap
- [x] Full Pipeline over a known dataset with specified camera intrinsics
- [x] Compatible dataset creation from a video with specified camera intrinsics
- [ ] Compatible dataset creation from a video with unspecified camera intrinsics(maybe using [this AnyCam paper](https://arxiv.org/pdf/2503.23282))

# Acknowledgments

- [**ConceptFusion**](https://arxiv.org/abs/2302.07241)
- [**CLIP-Fields**](https://arxiv.org/abs/2210.05663)
- [**LSeg**](https://arxiv.org/pdf/2201.03546)
- [**FastSAM**](https://github.com/CASIA-LMC-Lab/FastSAM)
- [**CLIP**](https://arxiv.org/pdf/2103.00020)
- [**MobileCLIP**](https://github.com/apple/ml-mobileclip)
- [**rerun.io**](https://rerun.io/)

