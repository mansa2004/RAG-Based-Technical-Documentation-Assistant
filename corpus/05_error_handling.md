# Error Handling

## Raising HTTPException

The standard way to return an error response with a specific status code is to raise `HTTPException`:

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
def read_item(item_id: str):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]
```

FastAPI catches `HTTPException` internally and converts it into a JSON response with the given status
code and a body of the form `{"detail": "Item not found"}`. Additional headers can be attached to the
error response via the `headers` parameter, which is useful for things like a `WWW-Authenticate` header
on a 401 response.

## Custom Exception Handlers

For application-specific error types, a custom exception handler can be registered using the
`@app.exception_handler()` decorator. This lets you define a custom exception class and control exactly
how it is serialized into an HTTP response, decoupling business logic (which raises the custom exception)
from HTTP-specific formatting (which lives only in the handler).

```python
class ItemNotFoundError(Exception):
    def __init__(self, item_id: str):
        self.item_id = item_id

@app.exception_handler(ItemNotFoundError)
def item_not_found_handler(request, exc: ItemNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"message": f"Item {exc.item_id} not found"},
    )
```

## Overriding Default Validation Errors

FastAPI's default handler for `RequestValidationError` (raised automatically when incoming data fails
Pydantic validation) returns a 422 status code with a detailed list of validation errors. This default handler
can be overridden by registering a custom handler for `RequestValidationError`, for example to return a
simplified error format or a different status code such as 400.

## Status Codes

The `status` module provides named constants for HTTP status codes (e.g., `status.HTTP_404_NOT_FOUND`)
so that route decorators and exception calls can avoid using raw integer "magic numbers," improving
readability. The `status_code` parameter on a path operation decorator sets the default success status
code returned when the handler completes without raising an exception.

## Handling Errors from Background Tasks

Exceptions raised inside a `BackgroundTasks` callable are not automatically converted into an HTTP error
response, because the response has typically already been sent to the client by the time the background
task runs. Such errors should be logged and handled internally by the background task itself, for example
by writing a failure record to a database or alerting system.
