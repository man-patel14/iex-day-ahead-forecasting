# Dashboard image: replays a finished backtest (results/ must contain predictions.npz and results.json).
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-app.txt
COPY . .
EXPOSE 8600
CMD ["streamlit", "run", "app/dashboard.py", "--server.port", "8600", "--server.address", "0.0.0.0"]
