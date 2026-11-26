# On Windows, run the commands directly (see README); `make` is not installed by default.
.PHONY: install test run data api dashboard

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

run:            ## full walk-forward backtest on synthetic data -> results/
	python run_backtest.py

data:           ## write the CSV template -> data/synthetic_demo.csv
	python scripts/make_synthetic_csv.py

api:
	uvicorn app.api:app --reload

dashboard:
	streamlit run app/dashboard.py
