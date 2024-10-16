FROM python:3.10-slim

WORKDIR /opt/app
COPY . .
RUN pip install --no-cache-dir -r /opt/app/requirements.txt
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]