"""FastAPI server for GenCast local weather app."""
import os, json, asyncio, time
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Set JAX memory flags before importing forecast engine
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')

from forecast_engine import run_forecast, load_model

app = FastAPI(title='GenCast Weather')

STATIC_DIR = Path(__file__).parent / 'static'
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')

# In-memory job store
_jobs: dict[str, dict] = {}


@app.on_event('startup')
async def startup():
    # Load model in background so server starts fast
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, load_model)


@app.get('/', response_class=HTMLResponse)
async def index():
    html = (STATIC_DIR / 'index.html').read_text()
    return HTMLResponse(html)


@app.post('/api/run')
async def run(background_tasks: BackgroundTasks, members: int = 5):
    job_id = f"job_{int(time.time())}"
    _jobs[job_id] = {'status': 'running', 'progress': 0, 'total': members,
                     'started': time.time(), 'data': None}

    def execute():
        def cb(done, total, elapsed):
            _jobs[job_id]['progress'] = done
        try:
            result = run_forecast(num_members=members, progress_cb=cb)
            _jobs[job_id].update({'status': 'done', 'data': result})
        except Exception as e:
            _jobs[job_id].update({'status': 'error', 'error': str(e)})

    background_tasks.add_task(execute)
    return {'job_id': job_id}


@app.get('/api/status/{job_id}')
async def status(job_id: str):
    if job_id not in _jobs:
        return JSONResponse({'error': 'not found'}, status_code=404)
    job = _jobs[job_id]
    return {
        'status':   job['status'],
        'progress': job['progress'],
        'total':    job['total'],
        'elapsed':  round(time.time() - job['started'], 1),
        'error':    job.get('error'),
    }


@app.get('/api/forecast/{job_id}')
async def forecast(job_id: str):
    if job_id not in _jobs:
        return JSONResponse({'error': 'not found'}, status_code=404)
    job = _jobs[job_id]
    if job['status'] != 'done':
        return JSONResponse({'error': 'not ready', 'status': job['status']}, status_code=202)
    return JSONResponse(job['data'])


if __name__ == '__main__':
    uvicorn.run('server:app', host='0.0.0.0', port=8000, reload=False)
