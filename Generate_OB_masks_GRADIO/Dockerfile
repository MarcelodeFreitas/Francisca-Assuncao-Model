# Dockerfile, Image, Container

FROM python:3.7.7

ADD deploy.py .

ADD requirements.txt .

ADD modelo_CNN_R_treinado_v4_174x190.h5 .
ADD modelo_CNN_treinado_v4.h5 .
ADD modelo_FCN_segmentation_32x32.h5 .

RUN pip install -r "./requirements.txt"

CMD ["python", "./deploy.py"]