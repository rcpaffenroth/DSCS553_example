FROM python:3.10-slim
# Can happen early, almost never changes
WORKDIR /opt/app

# Install packages that we need.
# Again does not change often 
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get upgrade -yq ca-certificates && \
    apt-get install -yq --no-install-recommends \
    prometheus-node-exporter

# Some environment variables that don't change often
EXPOSE 7860
EXPOSE 8000
EXPOSE 9100
ENV GRADIO_SERVER_NAME="0.0.0.0"

# We put this near the end since it can change
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

# This changes the most, so it goes last!
COPY . .
CMD bash -c "prometheus-node-exporter --web.listen-address=':9100' & python /opt/app/app.py"
