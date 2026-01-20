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

from fastapi.responses import HTMLResponse

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

3) RedirectResponse

RedirectResponse의 경우 크게 `Get` -> `Get` or `Post` -> `Get` 방식이 있다.
이때, 많이 쓰이는 방식은 `Post` -> `Get` 방식으로 대표적으로 로그인 이후 메인 페이지로 돌아가는 게 있다

**Get -> Get 으로 갈 때**

```python


from fastapi.responses import RedirectResponse
@app.get("/redirect")
async def redirect_only(comment: str | None = None):
    print(f"redirect {comment}")

    return RedirectResponse(url=f"/resp_html/3?item_name={comment}")

# 이떄 기본적으로 RedirectResponse의 status code는 307이다.

```

**Post -> Get으로 갈 때(Http method 변경)**

```python
@app.post("/create_redirect")
async def create_item(item_id: int = Form(), item_name: str = Form()):
    print(f"item_id: {item_id} item name: {item_name}")

    return RedirectResponse(url=f"/resp_html/{item_id}?item_name={item_name}", status_code=satus.HTTP_302_FOUND)

# 일반적으로 Http Method가 그대로이면 상관없지만 바뀔때에는
# status code를 307이 아니라 302로 해야지만 http method error가 발생하지 않는다 

```

| status_code |  내용 |
| ----------- | ----- |
|   `200`     | **OK** |
|   `302`     | **Found**, 명시적으로 변환 X, 대부분 웹 브라우저에서 변환 가능 |
|   `302`     | **SEE Other**, 명시적으로 Get으로 변환 |
|   `307`     | **Temporay Redirect** | 


4) HTTP status Code 개요

| Status 계열  |  계열 내용 |
| ----------- | ----------- |
|  `2XX`     |   성공적으로 요청 수행  | 
|  `3XX`     |   추가적인 Redirection 요청 |
|  `4XX`     |   Client의 잘못된 요청등의 오류  |
|  `5XX`     |   Server의 오류 | 


| 주요 Status Code | 코드 설명 |
| ---------------- | -------- | 
|  `200 OK` | Get/Post 등의 Request를 성공적으로 수행  |
|  `201 Created` | 새로운 리소스를 성공적으로 생성(Post 등)  |
|  `204 No Content` | 요청을 성공적으로 수행, 어떤 content도 반환 X  |
|  `301 Moved Permanently` | 요청된 Resource가 새로운 URL로 영구 이동 |
|  `302 Found` | 요청 Resource가 일시적으로 이전, 거의 대부분의 브라우저들이 사실상 HTTP 메소드를 GET으로 변경  |
|  `303 See Other` | 302와 비슷하지만 무조건 get으로 변경   |
|  `307 Temporary Redirect` | 요청 Resource가 일시적으로 다른 URL 이전, HTTP 메소드 동일  |
|  `400 Bad Request` | Server가 client의 요구를 이해할 수가 없음  |
|  `401 Unauthorized` | 요청 시 인증 실패  |
|  `404 Not Found` | 요청한 Request 자원을 찾을 수 없음 |
|  `405 Method Not Allowed` | 잘못된 HTPP 메소드로 요청함  |
|  `432 Unprocessable Entity` | 요청 포맷은 맞지만, 문맥적인 해석 불가 |
|  `500 Internal Server Error` | 비정상적인 상황의 오류 발생  |
|  `503 Service Unavailable` | 과부하/장애/유지보수 이유로 잠시 서비스 불가  |

