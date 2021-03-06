# Streamlit-coronavirus-detection-app

[![Deploy to heroku.](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/actions/workflows/main.yml/badge.svg)](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/actions/workflows/main.yml)
&nbsp;[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
&nbsp;[![Heroku App Status](http://heroku-shields.herokuapp.com/streamlit-github)](https://streamlit-github.herokuapp.com/)
&nbsp;[![Create and publish a Docker image](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/actions/workflows/Dockerimage.yaml/badge.svg)](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/actions/workflows/Dockerimage.yaml)
&nbsp;[![python - 3.9](https://upload.wikimedia.org/wikipedia/commons/1/1b/Blue_Python_3.9_Shield_Badge.svg)](https://www.python.org/downloads/release/python-397/)
<br><br>[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/repository/docker/kdocker03/streamlit-coronavirus-detection-app/general)&nbsp;[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/kaushal000/streamlit-coronavirus-detection-app/main/src/app.py)
&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KszU9b3t-T_Ia5GNjiy_uuktOnydlEID#scrollTo=O2w9w1Ye_nk1)
&nbsp;[![Microsoft PowerPoint](https://img.shields.io/badge/Microsoft_PowerPoint-B7472A?style=for-the-badge&logo=microsoft-powerpoint&logoColor=white)](https://1drv.ms/p/s!Asia4jQGHnLM6mzcO4j0SJydJfTD?e=UP4awf)


A streamlit app to detect coronaviruses from gray scaled electron microscopic images as well as gray scaled video. 

## Documentation 

* [Run Locally](#run-locally)
* [Run on docker container](#to-run-the-app-on-docker-change-the-directory-to-src-if-not-already)
    - [Build docker image](#to-build-the-image-type )
    - [Pull docker image](#alternatively-you-can-download-the-already-prebuilt-docker-image-by-either-of-the-two-way)
* [Live demo](#live-demo)
    - [Uploading and detecting coronaviruses from image](#uploading-image-and-detect-coronaviruses-from-the-image-camera)
    - [Uploading video and detecting coronaviruses from the uploaded video](#for-video-video_camera) 
* [Downloading weights](#downloading-trained-weights)
* [Train and test](#training-and-testing)
* [Results](#results)
    - [Weghts comparisons](#comparison-of-weights-for-confidence-threshold-0.25-and-IOU-threshold-0.5-or-50%)
    - [Mean average prediction chart](#mean-average-prediction-chart-at-different-epochs-along-with-loss)
* [Presentation](#presentation)
* [Deployment](#deployment)
    - [Streamlit share](#streamlit-share)
    - [Heroku](#heroku) 
* [Reference links](#reference-links)
* [Citation](#citation)



## Run Locally

Clone the project

```bash
  git clone https://github.com/Kaushal000/Streamlit-coronavirus-detection-app.git
```


Install dependencies

```bash
  pip install requirements.txt
```

Go to the src directory

```bash
  cd src
```

Start the server

```bash
  streamlit run app.py
```

On running the app it will automatically open your default browser and the app will be running on localhost:8501. If you browser doesn't open automatically open your browser and type the url in the adress bar 

```bash 
localhost:8501
```

## To run the app on docker change the directory to src if not already

### To build the image type 
```bash
docker image build -t streamlit-coronavirus-detection-app:latest`
```
**After building the image run the image on a container** 
```bash
docker run -p 8501:8501 streamlit-coronavirus-detection-app:latest
```

**To run it in background type** 
```bash
docker run -p 8501:8501 -d streamlit-coronavirus-detection-app:latest
```
### Alternatively you can download the already prebuilt docker image by either of the two ways 

```bash
docker pull ghcr.io/kaushal000/streamlit-coronavirus-detection-app:main
```
**or**

```bash
docker pull kdocker03/streamlit-coronavirus-detection-app
```

## Live demo

### Uploading image and detect coronaviruses from the image :camera:  
![Image](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/blob/main/Demo.gif)

### For video :video_camera:
![Video](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/blob/main/Demovideo.gif)
<br>Sample video to test [![GD](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=393665)](https://drive.google.com/file/d/1kVfiah1S-eiRa86webkjT_fYuryulmHh/view?usp=sharing)



## Downloading trained weights

**This app is automatically configured to download the best weight used for detection when this app runs for the first time if the model is not present inside the model folder.

***However you can download the best weight manually before running the app from here and put it inside the model folder*** 
:point_right: [![Dropbox](https://img.shields.io/badge/Dropbox-%233B4D98.svg?style=for-the-badge&logo=Dropbox&logoColor=white)](https://www.dropbox.com/s/909wlai4r3y4uz1/cov_yolov4_best.weights?dl=0)

**To download all the weights obtained at various epochs including the best one here is the link** 
:point_right: [![GD](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=393665)](https://drive.google.com/drive/folders/1nXGd1WZOlzk8fW36ADKfcHPvm3lBY7OT?usp=sharing)

## Training and testing 
Download the required files for training [![GD](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=393665)](https://drive.google.com/drive/folders/14Mrj9PmPaECouSZhlkoRIjiTF0FZSqUP?usp=sharing)

**The model is trained on a dataset containing 200 samples of grayscaled electron microscope coronavirus images and split into 80:20 train:test data which is for training there were 160 samples and for validation 20 samples** . An ipynb notebook containing the entire traing and testing process including the compilattion of darknet framwork is provided here <a href="https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/blob/main/Coronavirus_detection_train_and_test_model.ipynb">here</a>

Alternatively open in google colab to go through the traing and testing process here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KszU9b3t-T_Ia5GNjiy_uuktOnydlEID)

Read more about how to train custom models here [![colab](https://user-images.githubusercontent.com/4096485/86174097-b56b9000-bb29-11ea-9240-c17f6bacfc34.png)](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg?style=for-the-badge&logo=googledrive&logoColor=)

## Results

### Comparison of weights for confidence threshold 0.25 and IOU threshold 0.5 or 50%

| Weights                    | Precision | Recall | F1-score | map     | TP     | FP    | FN    |   
| :-----------------------:  | :-------: | :----: |  :----:  | :-----: | :----: | :---: | :---: |
| `cov_yolov4_1000.weights`  | `0.87`    | `0.89` |  `0.88`  | `89.13%`|  `150` |  `40` |  `25` | 
| `cov_yolov4_2000.weights`  | `0.93`    | `0.80` |  `0.86`  | `85.07%`|  `140` |  `11` |  `35` | 
| `cov_yolov4_best.weights`  | `0.87`    | `0.89` |  `0.88`  | `91.13%`|  `155` |  `23` |  `20` | 





### Mean average prediction chart at different epochs along with loss
![MAP](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/blob/main/map-chart/chart_cov_yolov4.png)

## Presentation 
The presentation for the entire project can be found here ???? [![Microsoft PowerPoint](https://img.shields.io/badge/Microsoft_PowerPoint-B7472A?style=for-the-badge&logo=microsoft-powerpoint&logoColor=white)](https://1drv.ms/p/s!Asia4jQGHnLM6mzcO4j0SJydJfTD?e=UP4awf)

## Deployment
### Streamlit share
**The app is deployed on streamlit share** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/kaushal000/streamlit-coronavirus-detection-app/main/src/app.py)

### Heroku
**The app is also deployed on heroku using Docker image with CI/CD enabled with the help of github actions** [![Heroku](https://img.shields.io/badge/heroku-%23430098.svg?style=for-the-badge&logo=heroku&logoColor=white)](https://streamlit-github.herokuapp.com/)

## Reference links
<a href="https://github.com/AlexeyAB/darknet" target="_blank">AlexeyAB/darknet</a>

Yolov4 paper ???? [![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2004.10934-B31B1B.svg)](https://arxiv.org/abs/2004.10934)
Scaled Yolov4 paper ???? [![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2011.08036-B31B1B.svg)](https://arxiv.org/abs/2011.08036)

<a href="https://baike.baidu.com/item/2019" target="_blank">Image, B.: The electron microscopic image of SARS-CoV-2</a> 

## Citation 
```
@misc{bochkovskiy2020yolov4,
      title={YOLOv4: Optimal Speed and Accuracy of Object Detection}, 
      author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
      year={2020},
      eprint={2004.10934},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
    title     = {{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {13029-13038}
}
```
