FROM python:3.10-slim

WORKDIR /opt/app
COPY . .
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

# Install packages that we need. vim is for helping with debugging
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get upgrade -yq ca-certificates && \
    apt-get install -yq --no-install-recommends \
    node-exporter

EXPOSE 7860
EXPOSE 8000
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
