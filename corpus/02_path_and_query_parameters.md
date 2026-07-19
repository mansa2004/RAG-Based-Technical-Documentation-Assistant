# Path and Query Parameters

## Path Parameters

Path parameters are declared using curly braces in the route string and matched by name to a function
argument:

```python
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

Because `item_id` is annotated as `int`, a request to `/items/abc` returns an automatic HTTP 422
Unprocessable Entity error with details about the validation failure, without any custom code being
written. A request to `/items/3` returns `item_id` as the integer `3`, not the string `"3"`.

## Order Matters for Fixed and Dynamic Paths

When a route has both a fixed path segment and a dynamic path parameter that could match it, the fixed
route must be declared first:

```python
@app.get("/users/me")
def read_current_user():
    return {"user": "current"}

@app.get("/users/{user_id}")
def read_user(user_id: str):
    return {"user_id": user_id}
```

If `/users/{user_id}` were declared first, a request to `/users/me` would be captured by that dynamic
route instead, with `user_id` set to the string `"me"`.

## Query Parameters

Function parameters that are not part of the path are automatically treated as query parameters. They can
have default values, making them optional:

```python
@app.get("/items/")
def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

A request to `/items/?skip=20` returns `skip=20` and `limit=10` (the default). Parameters without a
default value are treated as required, and a missing required query parameter results in a validation
error response.

## Combining Path and Query Parameters

Path and query parameters can be mixed freely in the same function signature; FastAPI determines which
is which based on whether the parameter name appears in the path template string.

## Using Enums to Restrict Values

To restrict a path or query parameter to a fixed set of valid values, a Python `Enum` class can be used as
the type annotation. FastAPI will validate the incoming value against the enum members and reflect the
allowed values in the generated documentation as a dropdown/select field.
