# PatternTime

**Local time-series investigation and performance regression analysis for developers**

PatternTime is a locally runnable Python tool for investigating application performance and telemetry over time. It helps developers find significant changes in metrics, measure statistical impact before and after a change, profile API routes, and evaluate whether a candidate build looks like a performance regression.

It is designed for local use with CSV datasets or ingested JSON telemetry. Detection uses statistical algorithms (Sliding Window, CUSUM, Robust CUSUM, and EWMA). PatternTime does **not** call paid AI APIs.

---

## Key Features

Verified from the current codebase:

- **CSV validation** — Load and validate timestamped metric files (`patterntime validate`).
- **Change-point detection** — Sliding Window, CUSUM, Robust CUSUM, and EWMA (`patterntime detect`).
- **Statistical impact analysis** — Pre/post means, Cohen’s *d*, severity, and summary sentences for detected changes.
- **Dataset registry** — Persist dataset metadata in MySQL (or SQLite via `DATABASE_URL`), including optional `application` and `environment` identity fields.
- **Dataset filtering** — List datasets filtered by `--application` and/or `--environment` (CLI table and `--json`).
- **Detection run history** — Save, list, show, and delete detection runs and related change events / impact reports.
- **Per-endpoint profiling** — Group by route/API name, compute latency percentiles and error rates, and rank potential regressions (`patterntime profile`).
- **CI/CD quality gate** — Compare candidate telemetry to a baseline or in-series thresholds; exit `0` pass / `1` fail (`patterntime gate`).
- **Telemetry ingestion** — Accept JSON batches via CLI (`patterntime ingest`) or HTTP (`POST /api/telemetry`), write CSV under `data/`, optionally run detection.
- **Local HTTP API + dashboards** — Standard-library server serves REST endpoints and static visualization pages (Apache ECharts).

---

## Technology Stack

| Area | Technology |
|---|---|
| Language / package | Python 3.10+ (`requires-python = ">=3.10"`) |
| Data / stats | NumPy, pandas, SciPy, python-dateutil |
| Persistence | SQLAlchemy 2.x, PyMySQL, python-dotenv, cryptography |
| CLI / API server | `patterntime` console script; `http.server`-based local server |
| Frontend charts | Apache ECharts + PapaParse (via `npm` local install, with CDN fallback) |
| Tests | pytest |

There is no required cloud or paid model dependency.

---

## Project Structure

```text
patterntime/
├── src/patterntime/          # Main Python package
│   ├── cli.py                # CLI entry point (all subcommands)
│   ├── server.py             # Local HTTP API + static file server
│   ├── storage.py            # SQLAlchemy models and DatabaseStorage
│   ├── detectors.py          # Sliding Window, CUSUM, Robust CUSUM, EWMA
│   ├── impact.py             # Statistical impact / effect-size analysis
│   ├── validator.py          # CSV / DataFrame validation
│   ├── telemetry.py          # Telemetry parse, ingest, query helpers
│   ├── profiler.py           # Per-endpoint profiling
│   ├── gate.py               # CI/CD quality-gate evaluation
│   ├── explore_data.py       # Utility script for dataset exploration
│   ├── evaluate_detector.py  # Utility script for detector evaluation
│   └── generate_*.py         # Sample / scenario data generators
├── visualization/            # Dashboards (index.html, executive.html, storage.js)
├── data/                     # Sample and evaluation CSV datasets
├── tests/                    # pytest suite
├── pyproject.toml            # Package metadata and console script
├── requirements.txt          # Pinned / listed runtime dependencies
├── package.json              # Frontend deps (echarts, papaparse)
└── .env.example              # Example DB environment variables
```

Virtual environments, caches, and generated files (for example `.venv/`, `node_modules/`, `.pytest_cache/`) are intentionally omitted here.

---

## Prerequisites

- **Python 3.10 or newer** (project metadata requires `>=3.10`; development here commonly uses 3.12)
- **Git**
- **A modern web browser** (for dashboards)
- **MySQL** — required when using the default MySQL connection settings for persistence
- **Node.js + npm** — recommended so dashboards can load ECharts/PapaParse from `node_modules/` (CDN fallback exists if local packages are missing)

Optional for local experimentation without MySQL: set `DATABASE_URL` to a SQLite URL (see [Database Setup](#database-setup)).

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/falgunisardhara/patterntime.git
cd patterntime
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies and the CLI

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` registers the `patterntime` console script from `pyproject.toml`.  
If you see `patterntime: command not found`, either activate `.venv` or call `.venv/bin/patterntime` directly.

You can also run the CLI as a module:

```bash
python -m patterntime --help
```

### 4. Install frontend dependencies (recommended)

```bash
npm install
```

This installs `echarts` and `papaparse` into `node_modules/`. The dashboards prefer those local files and fall back to CDN copies if they are unavailable.

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your local database settings. Example keys from `.env.example`:

```bash
DB_HOST=localhost
DB_PORT=3306
DB_NAME=patterntime
DB_USER=root
DB_PASSWORD=

# Optional full URL override (useful for SQLite in tests/local experiments):
# DATABASE_URL=mysql+pymysql://root:password@localhost:3306/patterntime
```

**Do not commit `.env` or real credentials.** `.gitignore` already ignores `.env` and `.env.*` while keeping `.env.example`.

Connection resolution order in code (`get_db_url()`):

1. `DATABASE_URL` if set  
2. Otherwise a composed `mysql+pymysql://...` URL from `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, and `DB_NAME` (with defaults)

---

## Database Setup

PatternTime persists datasets, detection runs, change events, and impact reports through SQLAlchemy models in `src/patterntime/storage.py`.

Tables:

| Table | Purpose |
|---|---|
| `datasets` | Dataset registry (`name`, `application`, `environment`, metrics metadata) |
| `detection_runs` | Saved detection executions |
| `change_events` | Individual detected change points |
| `impact_reports` | Pre/post statistical impact for a run |

Dataset identity is the composite unique constraint `uq_dataset_identity` on `(name, application, environment)`. The same filename can therefore exist for different application/environment combinations.

### Fresh database (new MySQL schema)

1. Create an empty MySQL database (example):

```sql
CREATE DATABASE patterntime CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

2. Configure `.env` (or `DATABASE_URL`) to point at that database.

3. Initialize tables:

```bash
patterntime db init
```

`db init` calls SQLAlchemy `create_all()`, which **creates missing tables**. It does **not** alter or migrate an existing table schema.

4. Verify connectivity and status:

```bash
patterntime db test
patterntime db status
```

### Using SQLite instead of MySQL (local / tests)

Set an explicit URL before running commands:

```bash
export DATABASE_URL="sqlite:///./patterntime_local.db"
patterntime db init
patterntime db test
```

### Existing database migration (application / environment columns)

If you already have a `datasets` table created from an older schema **without** `application` / `environment` (and without `uq_dataset_identity`), `patterntime db init` will **not** add those columns or the new unique constraint automatically.

Apply a manual migration on MySQL, then verify with `patterntime db status`.

Example migration steps (review and adapt to your live schema first):

```sql
-- 1) Add metadata columns if missing
ALTER TABLE datasets
  ADD COLUMN application VARCHAR(255) NULL,
  ADD COLUMN environment VARCHAR(100) NULL;

-- 2) Helpful indexes (names may vary if already present)
CREATE INDEX ix_datasets_application ON datasets (application);
CREATE INDEX ix_datasets_environment ON datasets (environment);

-- 3) If an older unique constraint existed on `name` alone, drop it first.
--    Inspect with: SHOW INDEX FROM datasets;
-- ALTER TABLE datasets DROP INDEX <old_unique_index_name>;

-- 4) Add composite identity constraint used by the current model
ALTER TABLE datasets
  ADD CONSTRAINT uq_dataset_identity UNIQUE (name, application, environment);
```

After migration:

```bash
patterntime db test
patterntime datasets list
```

> There is currently **no built-in migration command** in the CLI. Treat the SQL above as a manual checklist derived from `DatasetModel`, not as an automated migrator.

---

## Running the Application

Start the local API and static file server:

```bash
patterntime serve
```

Defaults from the CLI/server implementation:

- **Host:** `0.0.0.0`
- **Port:** `8089`

Custom bind:

```bash
patterntime serve --host 127.0.0.1 --port 8088
```

Equivalent module form:

```bash
python -m patterntime.server --port 8089
```

When the server starts it prints dashboard and status URLs. Open:

| Page | URL (default port 8089) |
|---|---|
| Engineering lab | http://127.0.0.1:8089/visualization/index.html |
| Executive dashboard | http://127.0.0.1:8089/visualization/executive.html |
| DB status API | http://127.0.0.1:8089/api/db/status |
| Root redirect | http://127.0.0.1:8089/ → engineering lab |

Convenience redirects also exist for `/workbench`, `/lab`, `/index.html`, `/executive`, and `/executive.html`.

**Stop the server:** press `Ctrl+C` in the terminal (the server catches `KeyboardInterrupt` and shuts down).

---

## Using the CLI

Show help:

```bash
patterntime --help
patterntime detect --help
```

### Database

```bash
patterntime db test
patterntime db init
patterntime db status
```

### Validate a CSV

```bash
patterntime validate data/sample_api_metrics.csv
patterntime validate data/sample_api_metrics.csv --json
```

### Detect changes

```bash
patterntime detect data/sample_api_metrics.csv \
  --metric response_time_ms \
  --algorithm sliding_window \
  --threshold 1.5 \
  --application my-laravel-app \
  --environment staging
```

Important arguments:

| Argument | Meaning |
|---|---|
| `--metric` | Metric column to analyze |
| `--algorithm` | `sliding_window` (default), `cusum`, `robust_cusum`, or `ewma` |
| `--window-size` | Sliding-window size (default `30`) |
| `--threshold` | Detection threshold (default `2.0`) |
| `--application` / `--environment` | Dataset identity metadata when saving |
| `--save` / `--no-save` | Persist run to DB (default is save) |
| `--json` | Machine-readable output |

### List datasets (with filters)

```bash
patterntime datasets list
patterntime datasets list --application my-laravel-app
patterntime datasets list --environment staging
patterntime datasets list --application my-laravel-app --environment staging
patterntime datasets list --application my-laravel-app --json
```

Table output includes `APPLICATION` and `ENVIRONMENT` columns.

### Inspection of saved runs

```bash
patterntime runs list
patterntime runs list --dataset sample_api_metrics.csv --limit 10
patterntime runs show 1
patterntime runs show 1 --json
```

### Profile API routes

```bash
patterntime profile data/user_activities.csv --min-samples 50
patterntime profile data/user_activities.csv --group-by api_name --metric execution_time_ms --json
```

### Quality gate

```bash
patterntime gate data/change_size_70ms.csv \
  --baseline data/no_change_metrics.csv \
  --metric response_time_ms \
  --format text
```

Exit codes: `0` if the gate passes, `1` if it fails, `2` on setup/evaluation errors.

### Ingest telemetry JSON

```bash
patterntime ingest path/to/batch.json --dataset api_telemetry.csv --mode append
patterntime ingest path/to/batch.json --detect --algorithm ewma --json
```

Payload formats are documented in [HTTP API and Telemetry](#http-api-and-telemetry).

---

## HTTP API and Telemetry

Base URL example: `http://127.0.0.1:8089`

### Endpoints (implemented in `server.py`)

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/db/status` | Connection check and table list |
| `GET` | `/api/datasets` | List datasets from the DB registry |
| `GET` | `/api/runs` | List runs (`?dataset=`, `?limit=`) |
| `GET` | `/api/runs/{id}` | Full run detail (events + impact reports) |
| `POST` | `/api/runs` | Persist a detection run payload |
| `DELETE` | `/api/runs/{id}` | Delete a run |
| `GET` | `/api/profile` | Profile a CSV under `data/` (`?dataset=`, `?group_by=`, `?metric=`, `?algorithm=`, `?min_samples=`) |
| `POST` | `/api/telemetry` | Ingest a telemetry batch |
| `GET` | `/api/telemetry` | List telemetry CSVs, or slice one dataset |

Notes verified in code:

- `GET /api/datasets` currently lists all registered datasets; it does **not** accept application/environment query filters (filtering is available on the CLI).
- CORS is permissive for local development (`Access-Control-Allow-Origin: *`).

### Example: database status

```bash
curl http://127.0.0.1:8089/api/db/status
```

### Example: persist a detection run

```bash
curl -X POST http://127.0.0.1:8089/api/runs \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "sample_api_metrics.csv",
    "metric_name": "response_time_ms",
    "algorithm_name": "sliding_window",
    "params": {"window_size": 30},
    "execution_time_ms": 3.1,
    "row_count": 100,
    "application": "my-laravel-app",
    "environment": "staging",
    "events": [
      {"index": 72, "direction": "up", "score": 10.5, "metadata": {}}
    ]
  }'
```

### Telemetry payload formats (`parse_telemetry_payload`)

**1. Batch envelope (recommended)**

```json
{
  "dataset": "gateway.csv",
  "service": "api-gateway",
  "application": "my-laravel-app",
  "environment": "staging",
  "mode": "append",
  "detect": false,
  "algorithm": "ewma",
  "metric": "latency_ms",
  "records": [
    {"timestamp": "2026-11-01 10:00:00", "route": "/pay", "latency_ms": 120.5, "status": 200},
    {"timestamp": "2026-11-01 10:00:05", "route": "/pay", "latency_ms": 118.0, "status": 200}
  ]
}
```

Supported record-list keys in the envelope: `records`, `data`, `metrics`, `points`, or `events`.

`mode` may be `append` (default) or `overwrite`.  
If `detect` is `true` and the dataset has at least 10 rows, optional change detection may run and can persist a run when DB storage is available.

**2. Raw list of records**

```json
[
  {"timestamp": "2026-11-01 10:00:00", "latency_ms": 50},
  {"timestamp": "2026-11-01 10:00:05", "latency_ms": 55}
]
```

Defaults to dataset name `api_telemetry.csv`.

**3. Single record object**

```json
{
  "timestamp": "2026-11-01 10:00:00",
  "latency_ms": 50,
  "application": "my-laravel-app",
  "environment": "staging",
  "dataset": "checkout.csv"
}
```

`application`, `environment`, and `dataset` are treated as options/metadata, not metric columns.

### Example: ingest and query telemetry

```bash
curl -X POST http://127.0.0.1:8089/api/telemetry \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "demo_ingest.csv",
    "application": "my-laravel-app",
    "environment": "staging",
    "service": "billing-api",
    "records": [
      {"timestamp": "2026-11-01 10:00:00", "route": "/charge", "duration_ms": 140.5, "status": 200},
      {"timestamp": "2026-11-01 10:00:05", "route": "/charge", "duration_ms": 145.2, "status": 200}
    ]
  }'

curl "http://127.0.0.1:8089/api/telemetry?dataset=demo_ingest.csv&limit=10"
curl "http://127.0.0.1:8089/api/telemetry?dataset=demo_ingest.csv&route=/charge"
curl http://127.0.0.1:8089/api/telemetry
```

Ingested rows are written as CSV files under `data/`. Dataset metadata may also be stored in the database when storage is available.

---

## Visualization

With `patterntime serve` running:

### Engineering lab — `/visualization/index.html`

Interactive investigation UI for:

- Selecting or uploading CSV datasets
- Running change detection and viewing charts (ECharts)
- Reviewing impact summaries
- Opening MySQL run history (when the DB is configured)
- Profiling API routes from suitable datasets (for example `user_activities.csv`)

### Executive dashboard — `/visualization/executive.html`

Stakeholder-oriented summary view of reliability / anomaly context, with a link back to the engineering lab.

Both pages load ECharts and PapaParse from `../node_modules/...` first, then CDN fallbacks.

---

## Testing

Install pytest if it is not already in your environment:

```bash
pip install pytest
```

Run the suite from the repository root (with the venv active and the package installed):

```bash
pytest -q
```

Or:

```bash
.venv/bin/pytest -q
```

`pyproject.toml` configures `pythonpath = ["src"]` and `testpaths = ["tests"]`.

Focused examples:

```bash
pytest tests/test_storage.py::test_list_datasets -v
pytest tests/test_cli.py::test_cli_datasets_list_filters -v
pytest tests/test_server.py -v
pytest tests/test_telemetry.py -v
```

Many storage/server tests use in-memory or temporary SQLite via `DATABASE_URL` and do not require a live MySQL instance.

---

## Troubleshooting

| Problem | What to check |
|---|---|
| `patterntime: command not found` | Activate `.venv`, or use `.venv/bin/patterntime`, or run `pip install -e .` |
| MySQL connection failures | Verify `.env` / `DATABASE_URL`, that MySQL is running, and that the database exists; run `patterntime db test` |
| `db init` succeeds but filters / metadata columns are missing | Existing tables are not auto-migrated; apply the [manual migration](#existing-database-migration-application--environment-columns) |
| Dashboards show blank charts / missing libraries | Run `npm install`, or confirm CDN access for the fallback scripts |
| Port already in use | Choose another port: `patterntime serve --port 8088` |
| Telemetry POST returns 400 | Ensure the payload has a non-empty `records` list (or valid list/single-record form) and at least one numeric metric |
| Profile/gate on `user_activities.csv` fails validation in `detect` | That file includes missing values in `status_code`; prefer `profile` / `gate` flows or cleaned CSVs for `detect`/`validate` |
| Ambiguous dataset name errors | Provide `--application` and `--environment` (or a dataset id) when multiple rows share the same filename |

---

## Limitations and Future Improvements

### Implemented today

- Local statistical detection, impact analysis, profiling, quality gate, CLI, HTTP API, dashboards, and SQLAlchemy persistence
- Application/environment dataset identity and CLI filtering
- Telemetry batch ingestion to CSV (+ optional detection)

### Not implemented / planned (do not treat as available)

- Automatic schema migrations / upgrade command for existing databases
- HTTP query filters for `GET /api/datasets` by application/environment
- Paid or generative AI detection backends
- A bundled Laravel (or other app) agent for live metric export (mentioned historically as future work)
- Guaranteeing that every detector finds every change on every dataset

PatternTime is intended for local investigation with synthetic or sanitized telemetry. Review bind address, CORS, and credentials before exposing the server beyond your machine.

---

## Demo Workflow

A short first-run path using sample data already in the repo:

```bash
# 0) Setup (once)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install pytest
npm install
cp .env.example .env
# Edit .env for MySQL, or:
# export DATABASE_URL="sqlite:///./patterntime_demo.db"

# 1) Initialize and verify DB
patterntime db init
patterntime db test

# 2) Validate + detect on sample metrics
patterntime validate data/sample_api_metrics.csv
patterntime detect data/sample_api_metrics.csv \
  --metric response_time_ms \
  --algorithm sliding_window \
  --threshold 1.5 \
  --application demo-app \
  --environment local

# 3) Inspect registry and runs
patterntime datasets list --application demo-app
patterntime datasets list --application demo-app --environment local --json
patterntime runs list
patterntime runs show 1

# 4) Profile a multi-route dataset
patterntime profile data/user_activities.csv --min-samples 50

# 5) Optional quality gate comparison
patterntime gate data/change_size_70ms.csv \
  --baseline data/no_change_metrics.csv \
  --metric response_time_ms

# 6) Start UI + API
patterntime serve
# Open http://127.0.0.1:8089/visualization/index.html
# Open http://127.0.0.1:8089/visualization/executive.html
```

Optional telemetry ingest while the server is running:

```bash
curl -X POST http://127.0.0.1:8089/api/telemetry \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "demo_ingest.csv",
    "application": "demo-app",
    "environment": "local",
    "records": [
      {"timestamp": "2026-11-01 10:00:00", "latency_ms": 110},
      {"timestamp": "2026-11-01 10:00:05", "latency_ms": 112}
    ]
  }'
```

---

## Sample Datasets

Files under `data/` include:

| File | Typical use |
|---|---|
| `sample_api_metrics.csv` | General detect / validate demo (`response_time_ms`) |
| `user_activities.csv` | Route profiling (`api_name`, `execution_time_ms`) |
| `no_change_metrics.csv` | Stable baseline / evaluation |
| `change_size_*.csv` | Controlled magnitude-change scenarios |
| `scenario_*.csv` | Noise, drift, and spike scenarios |

---

## License / Repository

Source repository: [https://github.com/falgunisardhara/patterntime](https://github.com/falgunisardhara/patterntime)
