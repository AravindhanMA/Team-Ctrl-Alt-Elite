FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT (defaults to 8080) and expects the container to
# listen on 0.0.0.0 -- both required, or the deploy will fail health checks.
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

CMD streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0
