FROM nvcr.io/nvidia/tensorflow:19.12-tf2-py3

LABEL maintainer="Timothy Liu <timothyl@nvidia.com>"

USER root

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]
