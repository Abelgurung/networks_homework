FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

COPY iperf3_client.py goodput_measure.py goodput_plot.py \
     tcp_stats_measure.py tcp_stats_plot.py ml_plot.py \
     run_all.sh ./
COPY ML_model/ ./ML_model/

RUN sed -i 's/\r$//' run_all.sh && chmod +x run_all.sh

CMD ["./run_all.sh"]
