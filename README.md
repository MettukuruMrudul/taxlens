# TaxLens

Tax-year-aware, citation-grounded Q&A over IRS forms, instructions, and publications.

> TaxLens provides information from IRS publications for educational purposes only. It is not tax advice.

## Status

Week 1 of the build plan: local infrastructure and the document manifest for the MVP corpus
(Form 1040 family + Publication 17, tax years 2021–2024).

## Quickstart (Windows PowerShell)

```powershell
docker compose up -d                     # start Postgres (pgvector) and Redis
python -m venv .venv                     # create a virtual environment
.\.venv\Scripts\Activate.ps1             # activate it
pip install -r requirements.txt          # install dependencies
copy .env.example .env                   # create local settings
python -m pytest                         # run tests
python -m taxlens.ingest.fetch           # download the MVP corpus and fill the manifest
```

## Project layout

```
db/init/            SQL that runs when the database is first created
taxlens/config.py   settings loaded from .env
taxlens/db.py       database connection helper
taxlens/ingest/     corpus definition and the fetcher
tests/              pytest tests
data/raw/           downloaded IRS PDFs (not committed)
```

## Note on schema changes

Files in `db/init/` only run when the database volume is empty. To rebuild the
database from scratch during early development: `docker compose down -v` then
`docker compose up -d`. This deletes all database data.
