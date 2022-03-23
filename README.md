# Streamlit-coronavirus-detection-app

[![Deploy to heroku.](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/actions/workflows/main.yml/badge.svg)](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/actions/workflows/main.yml)
&nbsp;[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
&nbsp;[![Heroku App Status](http://heroku-shields.herokuapp.com/streamlit-github)](https://streamlit-github.herokuapp.com/)
&nbsp;[![Create and publish a Docker image](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/actions/workflows/Dockerimage.yaml/badge.svg)](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/actions/workflows/Dockerimage.yaml)
&nbsp;[![python - 3.9](https://upload.wikimedia.org/wikipedia/commons/1/1b/Blue_Python_3.9_Shield_Badge.svg)](https://www.python.org/downloads/release/python-397/)
<br><br>[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/repository/docker/kdocker03/streamlit-coronavirus-detection-app/general)&nbsp;[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/kaushal000/streamlit-coronavirus-detection-app/main/src/app.py)
&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KszU9b3t-T_Ia5GNjiy_uuktOnydlEID)


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
    - [Mean average prediction chart](#mean-average-prediction-chart)



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


```bash
docker pull kdocker03/streamlit-coronavirus-detection-app
```

## Live demo

### Uploading image and detect coronaviruses from the image :camera:  
![Image](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/blob/main/Demo.gif)

### For video :video_camera:
![Video](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/blob/main/Demovideo.gif)



## Downloading trained weights
**This app is automatically configured to download the best weight used for detection when this app runs for the first time if the model is not present inside the model folder.

***However you can download the best weight manually before running the app from here and put it inside the model folder*** 
:point_right: [![Dropbox](https://img.shields.io/badge/Dropbox-%233B4D98.svg?style=for-the-badge&logo=Dropbox&logoColor=white)](https://www.dropbox.com/s/909wlai4r3y4uz1/cov_yolov4_best.weights?dl=0)

**To download all the weights obtained at various epochs including the best one here is the link** 
:point_right: [![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1nXGd1WZOlzk8fW36ADKfcHPvm3lBY7OT?usp=sharing)

## Training and testing 
**The model is trained on a dataset containing 200 samples of grayscaled electron microscope coronavirus images and split into 80:20 train:test data which is for training there were 160 samples and for validation 20 samples** . An ipynb notebook containing the entire traing and testing process is provided ![here]('Coronavirus_detection_train_and_test_model.ipynb').

Alternatively open in google colab to go through the traing and testing process [here](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/blob/main/Coronavirus_detection_train_and_test_model.ipynb)

Read more about how to train custom models here [![colab](https://user-images.githubusercontent.com/4096485/86174097-b56b9000-bb29-11ea-9240-c17f6bacfc34.png)](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg)

## Results

### Mean average prediction chart
![MAP](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/blob/main/map-chart/chart_cov_yolov4.png)

