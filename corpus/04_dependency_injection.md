# Dependency Injection

## What Dependencies Are

FastAPI includes a dependency injection system that lets you declare shared logic (database connections,
authentication checks, pagination parameters, rate limiting, etc.) once and reuse it across many route
handlers. A dependency is simply a callable — usually a function — that FastAPI will call for you before
running the path operation function, injecting its return value into the parameter that declares it.

```python
from fastapi import Depends

def common_params(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

@app.get("/items/")
def read_items(params: dict = Depends(common_params)):
    return params
```

## Class-Based Dependencies

Dependencies can also be classes. FastAPI will instantiate the class using the same parameter-resolution
rules it applies to path operation functions, then inject the instance:

```python
class CommonQueryParams:
    def __init__(self, skip: int = 0, limit: int = 10):
        self.skip = skip
        self.limit = limit

@app.get("/items/")
def read_items(commons: CommonQueryParams = Depends()):
    return {"skip": commons.skip, "limit": commons.limit}
```

## Dependencies with Yield

A dependency function can use `yield` instead of `return` to define setup and teardown logic, which is the
standard pattern for managing resources like database sessions:

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

Code before the `yield` runs before the path operation; code after the `yield` runs after the response has
been sent, even if an exception occurred, making it a reliable place to release resources.

## Sub-Dependencies

Dependencies can themselves depend on other dependencies, and FastAPI resolves the whole chain. If
multiple path operations in the same request share a sub-dependency, FastAPI caches the result within a
single request by default, so the underlying callable is only executed once per request unless caching is
explicitly disabled.

## Dependencies at the Router or Application Level

Instead of adding `Depends()` to every path operation individually, dependencies can be declared on an
entire `APIRouter` or on the whole `FastAPI` app instance, which is a common pattern for enforcing
authentication across a whole group of routes without repeating the dependency declaration on every
endpoint.
