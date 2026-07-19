# FastAPI Basics

## What is FastAPI

FastAPI is a modern Python web framework for building APIs. It is built on top of Starlette (for the web
handling parts) and Pydantic (for data validation). It is designed to let developers write less code while
getting automatic request validation, serialization, and interactive documentation.

## Creating an Application

A minimal FastAPI application starts with creating an `app` instance and attaching route handlers to it:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}
```

Running this with `uvicorn main:app --reload` starts a local development server. FastAPI automatically
generates interactive API documentation at `/docs` (Swagger UI) and `/redoc` (ReDoc), derived from the
route definitions and type hints.

## Type Hints Drive Behavior

FastAPI relies heavily on standard Python type hints. When you annotate a function parameter, FastAPI
uses that annotation to:

1. Validate incoming data (rejecting requests that don't match the expected type)
2. Convert/coerce compatible data (e.g., a query string `"5"` becomes an `int` 5)
3. Generate accurate OpenAPI schema documentation
4. Provide editor autocompletion when writing handler code

This means the same type hints serve documentation, validation, and IDE support simultaneously, which is
a core productivity feature of the framework.

## ASGI Foundation

FastAPI is an ASGI (Asynchronous Server Gateway Interface) framework, meaning route handlers can be
defined as either regular `def` functions or `async def` functions. Using `async def` allows a single worker
process to handle many concurrent requests efficiently when the handler performs I/O-bound work (like
calling a database or an external API), because the event loop can switch to other requests while waiting.
CPU-bound work in an `async def` handler will still block the event loop, so heavy computation should
either run in a regular `def` function (FastAPI runs those in a thread pool automatically) or be offloaded to
a background worker.

## Running in Production

For production deployments, FastAPI applications are typically served with an ASGI server such as
Uvicorn or Hypercorn, often behind a process manager like Gunicorn using the `uvicorn.workers.UvicornWorker`
worker class to run multiple processes across CPU cores.
