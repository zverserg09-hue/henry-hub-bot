name: Henry Hub Signals

on:
  push:
    branches: ["main"]
  schedule:
    - cron: "0 9 * * *"
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest

    env:
      EIA_API_KEY: \${{ secrets.EIA_API_KEY }}

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run signal script
        run: python henry_hub_signals.py

