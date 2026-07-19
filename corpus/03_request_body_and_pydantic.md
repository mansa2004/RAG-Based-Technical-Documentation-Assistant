# Request Bodies and Pydantic Models

## Declaring a Request Body

To accept a JSON request body, define a class that inherits from Pydantic's `BaseModel`, then use it as a
type annotation on a path operation function:

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None

@app.post("/items/")
def create_item(item: Item):
    return item
```

FastAPI reads the request body as JSON, validates it against the `Item` model, converts matching fields to
their declared Python types, and passes the result into the function as a fully typed object. If the body is
missing a required field or has a field of the wrong type, a 422 response is returned automatically with a
structured error explaining exactly which field failed and why.

## Nested Models

Pydantic models can be nested inside other models, and FastAPI will validate the entire nested structure,
including lists of nested models:

```python
class Image(BaseModel):
    url: str
    name: str

class Item(BaseModel):
    name: str
    tags: list[str] = []
    image: Image | None = None
```

## Combining Body, Path, and Query Parameters

A single path operation function can accept a request body model, one or more path parameters, and one
or more query parameters simultaneously. FastAPI distinguishes them by: parameters found in the path
template are path parameters, parameters that are Pydantic models are treated as body fields, and
everything else with a simple type is treated as a query parameter.

## Field Validation and Defaults

The `Field` function from Pydantic allows adding extra validation and metadata to individual model fields,
such as minimum/maximum length for strings, numeric bounds, regex patterns, and example values shown
in the documentation:

```python
from pydantic import Field

class Item(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    price: float = Field(..., gt=0, description="Price must be greater than zero")
```

## Response Models

A separate response model can be declared using the `response_model` parameter of the path operation
decorator. This is useful for filtering out fields that should not be exposed in the API response, such as a
hashed password field that exists on the input model but should never be returned to a client.
