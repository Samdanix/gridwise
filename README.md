# GridWise

GridWise is a stateless Django REST API that optimizes hourly energy usage using solar generation, battery storage, grid limits, tariffs, and operator instructions.

## Live Demo

**https://gridwise-96tb.onrender.com/**

### API Endpoints

- `GET /health` — Health/readiness check
- `POST /optimize-energy` — Generate an optimized 24-hour energy plan
- `GET /diagnostics` — Local/deployment diagnostics

## Tech Stack

- Python
- Django
- Django REST Framework
- PuLP / CBC
- Google Gemini API
- Gunicorn
- Pytest

## Project Structure

```text
gridwise/
├── gridwise/          # Django project configuration
├── optimizer/         # Optimization, LLM, validation and API logic
├── optimizer/tests/   # Unit and integration tests
├── harness/           # Judge-style testing utilities
├── manage.py
├── requirements.txt
└── Dockerfile
```

## Setup

1. Clone the repository and enter the project directory.

```bash
cd gridwise
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. Create `.env` from `.env.example` and configure the required API key.

```env
GEMINI_API_KEY=your-google-ai-studio-key
```

5. Start the development server.

```bash
python manage.py runserver
```

The API will be available at `http://127.0.0.1:8000`.

## Testing

### Unit Tests

Run the complete automated test suite:

```bash
pytest
```

Run only optimizer tests:

```bash
pytest optimizer/tests
```

### API Testing

Check the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

For the deployed API:

```bash
curl https://gridwise-96tb.onrender.com/health
```

### Judge-Style Testing

It's straightforward — just one command:

```bash
python gridwise_test.py --url https://your-api-url.com --cases BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json
```

Replace `https://your-api-url.com` with whatever URL your API is running on. For example:

- Local: `--url http://localhost:8000`
- Deployed: `--url https://gridwise-abc123.onrender.com`

The `--cases` file needs to be in the same folder, or you can provide the full path to it.

**That's it.** The harness automatically runs all 10 public sample cases + 12 hidden-style paraphrase cases, the malformed-input suite, determinism checks, and latency probes. At the end, it provides a rubric estimate and a checklist.

For this deployment:

```bash
python gridwise_test.py --url https://gridwise-96tb.onrender.com --cases BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json
```

The harness checks items such as API response structure, directive interpretation, hourly-plan validity, energy/battery constraints, totals, optimization quality, malformed requests, stability, latency, and secret leakage.

## Notes

GridWise is designed to be stateless and does not require a database or user sessions. The optimizer produces a request/response energy schedule and validates the resulting plan against the applicable constraints.
