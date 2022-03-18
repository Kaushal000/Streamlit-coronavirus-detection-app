FROM python:3.9.7
EXPOSE 8501
COPY ./requirements_for_docker.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
COPY . /app
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
# CMD streamlit run --server.port $PORT app.py
