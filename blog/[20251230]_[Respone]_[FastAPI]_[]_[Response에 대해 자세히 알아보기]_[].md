## 1. HTTP Response 개요

- <ins/>HTTP Response는 client Request에 따른 server에서 보내는 메세지 </ins>
- <ins/>요청 Requset의 처리 상태, 여러 메타 정보, 그리고 Content 데이터를 담고 있음</ins>

> **Response 메세지 예시**

```bash
HTTP1.1 200 OK
Content-Type: text/html
Content-Length: 1234
Server: apache

<!DOCTYPE html>
<html>
...
</html>

```

- 1. Status-Code: HTTP Version과 Response 상태 코드
- 2. Response Header: Content Type, 다양한 메타 정보
- 3. Blank Line: Header와 Body를 구분
- 4. Response Body: HTML이나 JSON, Image 등의 client에게 전달되는 실질 데이터 

| Response Class Type | 설명 | 
| ------------------- | ---- |
|  `JSONResponse` | JSON 타입 Content 전송, Python object를 json format으로 자동 변환|
| `HTMLResponse` | HTML Content 전송 |
| `RedirectResponse` | 요청 처리 후 다른 URL로 Client를 다른 URL로 Redirect하기 위해 사용 |
| `PlainTextResponse` | 일반 text Content 전송 |
| `FileResponse` | 파일을 download하는데 주로 사용 |
| `StreamingResponse` | 대용량 파일의 Streaming 이나 chat message 등에 사용 | 

## 2. HTTP Respone Class

1) JSONResponse

원래 @app.get(path='/items')에서 path argument 말고도 다양한 argument가 존재한다. 그중 response_class라는 argument가 있고 기본적으로 default 값은 JSONResponse이다

```
"""
JSONResponse Parameter
"""
content: Any
status_code: int = 200
headers: Mapping[str, str] | None = None
media_type: str | None = None

```

2) HTMLResponse

```python
# HTML Response
@app.get("/resp_html/{item_id}", response_class=HTMLResponse)

async def response_html(item_id: int, item_name: str | None = None):
    html_str = f'''
    <html>
    <body>
        <h2>Response Body</h2>
        <p>item_id: {item_id}</p>
        <p>item_name: {item_name}</p>
    </body>
    </html>
    '''
    
    return HTMLResponse(html_str, status_code=status.HTTP_200_OK)    
```