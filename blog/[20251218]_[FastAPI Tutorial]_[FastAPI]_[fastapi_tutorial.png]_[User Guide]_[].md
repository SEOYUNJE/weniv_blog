## 1. HTTP Method 종류

| Http method | Description |
| ----------- | ----------- |
|    `GET`      |  **서버에서 특정 파일을 가져오도록 요청**   |
|    `POST`      |   **서버에서 특정 정보를 전달(데이터 및 파일)**    |
|    `PUT`     |  **서버에 특정 정보를 업데이트**     |
|    `DELETE`      |   **서버에 특정 정보를 삭제**      |
|    `PATCH`      |    **서버에 특정 정보를 업데이트**      |
|    `HEAD`      |   **서버에서 Head 정보만 가져오도록 요청**   |

## 2. Path Parameters

- <ins/>URL Path이 일부로서 path에 정보를 담아서 Requests로 전달</ins>
- <ins/>URL이 https://seoyunje.github.io/develop_blog/2라면 path는 develop_blog/2이면 path parameter는 2이다.</ins> 
- <ins/>메세지는 Body 없이 전달</ins>

**`default usage`**

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}
```

**`path parameters with types`**

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int) ## type hint
    return {"item_id": item_id}

```

## 3. Query Parameters

- <ins/>Query string이라고도 불리며 url에서 ?뒤에 key와 value 값을 가지는 형태로 requests로 전달, 개별 parameter는 &로 분리</ins>
- <ins/>https://seoyunje.github.io/develop_blog/skip=1&limit=2라면 skip, limit가 query parameter이며 각 parameter 값은 1, 2이다</ins> 

**`default usage`**

```python
from fastapi import FastAPI

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

app = FastAPI()

@app.get("/items/")
async def read_item(skip, limit):
    return fake_items_db[skip: skip + limit]
```

**`query parameters with type int`**

```python
from fastapi import FastAPI

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

app = FastAPI()

@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 2):
    return fake_items_db[skip: skip + limit]
```

**`query parameters with path parameters`**

```python
from fastapi import FastAPI

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int, skip: int = 0, limit: int = 2):
    return fake_items_db[skip: skip + limit]
```