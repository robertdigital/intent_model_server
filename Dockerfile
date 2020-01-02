FROM nvaitc/ai-lab:19.11-batch-tf2

LABEL maintainer="Timothy Liu <timothyl@nvidia.com>"

USER 1000

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]
