## FastAPI의 Request 객체

해당 객체는 HTTP Request에 대한 정보를 가지고 있는 객체로

```python
from fastapi import Requset

```

| request 내 field 유형 | 상세 설명 |
| --------------------- | -------- |
|   `request.method`    |         |
|   `request.url`    |         |
|   `request.header`    |         |
|   `request.client`    |         |
|   `request.query_param`    |         |
|   `request.json`    |         |
|   `request.form`    |         |


아래는 Display 예시입니다

```json
{
    "client_host": "127.0.0.1"
}
```

_**Default Usage**_

```python
from fastapi import Request

@app.get("/items")
async def read_item(request: Request):
    client_host = request.client.host
    headers = request.headers
    query_params = request.query_params
    url = request.url
    path_params = request.path_params
    http_method = request.method

    return {
        "client_host": client_host,
        "headers": headers,
        "query_params": query_params,
        "path_parmas": path_params,
        "url": str(url),
        "http_method": http_method
    }

```

_**Using Parse Json Body**_

_**Using Parse Form Body**_

