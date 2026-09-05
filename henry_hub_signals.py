name: Henry Hub Signals

on:
  schedule:
    # Запуск каждый день в 08:00, 12:00, 16:00, 20:00 UTC
    - cron: "0 8,12,16,20 * * *"
  # Можно запустить вручную кнопкой "Run workflow" в интерфейсе GitHub
  workflow_dispatch:

jobs:
  run-script:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run the signal script
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          EIA_API_KEY: ${{ secrets.EIA_API_KEY }}
          STORAGE_CURRENT_BCF: "3153"
          LAST_STORAGE_BUILD: "15"
          STORAGE_FORECAST: "19"
        run: python henry_hub_signals.py
