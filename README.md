# Streamlit-coronavirus-detection-app

[![Deploy to heroku.](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/actions/workflows/main.yml/badge.svg)](https://github.com/Kaushal000/Streamlit-coronavirus-detection-app/actions/workflows/main.yml)

A streamlit app to detect coronaviruses from gray scaled electron microscopic images as well as gray scaled video. 

##To run the app locally git clone the repository and cd into Streamlit-coronavirus-detection-app##

Install the dependencies by typing the command below in the terminal
`pip install -r requirements.txt`

Then run streamlit app by typing `streamlit run src/app.py`

On running the app it will automatically open your default browser and the app will be runing on localhost:8501. If you browser doesn't open automatically open your browser and type the url in the adress bar `localhost:8501`

##To run the app on docker change the directory to src by typing `cd src`##

To build the imgae type `docker image build -t streamlit-coronavirus-detection-app:latest`

After building the image run the image on a container by typing `docker run -p 8501:8501 streamlit-coronavirus-detection-app:latest`

To run it in background type `docker run -p 8501:8501 -d streamlit-coronavirus-detection-app:latest`

##Live app deployed on streamlit##
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/kaushal000/streamlit-coronavirus-detection-app/main/src/app.py)

##docker container deployed on heroku 

