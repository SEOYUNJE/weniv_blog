## 1. HTTP Method 종류

| Http method | Description |
| ----------- | ----------- |
|    `GET`      |  **서버에서 특정를 가져오도록 요청**   |
|    `POST`      |   **서버에서 특정 정보를 전달(데이터 및 파일)**    |
|    `PUT`     |  **서버에 특정 정보를 업데이트**     |
|    `DELETE`      |   **서버에 특정 정보를 삭제**      |
|    `PATCH`      |    **서버에 특정 정보를 업데이트**      |
|    `HEAD`      |   **서버에서 Head 정보만 가져오도록 요청**   |

## 2. Path Parameters

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