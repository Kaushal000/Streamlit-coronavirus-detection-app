# Streamlit-coronavirus-detection-app

[![Deploy to heroku.](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/actions/workflows/main.yml/badge.svg)](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/actions/workflows/main.yml)
&nbsp;[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
&nbsp;[![PyPI - Wheel](https://img.shields.io/pypi/wheel/streamlit)](https://streamlit.io/)
&nbsp;[![Heroku App Status](http://heroku-shields.herokuapp.com/streamlit-github)](https://streamlit-github.herokuapp.com/)
&nbsp;[![Create and publish a Docker image](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/actions/workflows/Dockerimage.yaml/badge.svg)](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/actions/workflows/Dockerimage.yaml)
&nbsp;[![python - 3.9](https://upload.wikimedia.org/wikipedia/commons/1/1b/Blue_Python_3.9_Shield_Badge.svg)](https://www.python.org/downloads/release/python-397/)
<br><br>[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/repository/docker/kdocker03/streamlit-coronavirus-detection-app/general)&nbsp;[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/kaushal000/streamlit-coronavirus-detection-app/main/src/app.py)
&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kaushal000/Streamlit-coronavirus-detection-app/blob/main/Coronavirus_detection_model_training.ipynb)


A streamlit app to detect coronaviruses from gray scaled electron microscopic images as well as gray scaled video. 

## Documentation 

* [Run Locally](#run-locally)
* [Run on docker container](#after-building-the-image-run-the-image-on-a-container)
    - [Build docker image](#to-run-the-app-on-docker-change-the-directory-to-src-if-not-already)
    - [Pull docker image](#alternatively-you-can-download-the-already-prebuilt-docker-image-by-either-of-the-two-way ) 
* [Downloading weights](#downloading-trained-weights)
* [Mean average prediction chart](#mean-average-prediction-chart)



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

### To run the app on docker change the directory to src if not already

To build the image type 
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

## Live demo to upload an image and detect coronaviruses from the image :point_down: 
![alt text](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/blob/main/Demo.gif)

## You can do similar step to upload a video and detect coronaviruses from the video

## Downloading trained weights
**This app is automatically configured to download the best weight used for detection when this app runs for the first time if the model is not present inside the model folder.

***However you can download the best weight manually before running the app from here and put it inside the model folder*** 
:point_right: [![Dropbox](https://img.shields.io/badge/Dropbox-%233B4D98.svg?style=for-the-badge&logo=Dropbox&logoColor=white)](https://www.dropbox.com/s/909wlai4r3y4uz1/cov_yolov4_best.weights?dl=0)

**To download all the weights obtained at various epochs including the best one here is the link** 
:point_right: [![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1nXGd1WZOlzk8fW36ADKfcHPvm3lBY7OT?usp=sharing)

## Mean average prediction chart
![MAP](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/blob/main/map-chart/chart_cov_yolov4.png)

