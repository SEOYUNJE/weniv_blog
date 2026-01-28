# Pydantic

- Schema/데이터 타입 검증 및 데이터 값에 대한 검증 수행
- 정규식 지원 및 다양한 내장 검증 로직 제공
- Core 검증 로직은 Rust로 제작되어 가장 빠른 파이썬 데이터 검증 라이브러리

- ex) from pydantic import BaseModel

- Pydantic 객체는 BaseModel을 상속한 Pydantic Model 클래스에 기반하여 생성. Pydantic Model은 클래스 속성들의 Type hint, Optional, Default 값등의 Schema 구조 및 Validation 로직을 선언함

## Pydantic 모델 상속

```python
from pydantic import BaseModel

## 첫번째 Pydantic 모델
class User(BaseModel):
    id: int
    name: str
    email: str
    age: int | None = None

## User를 상속받은 Pydantic 모델
class AdvancedUser(User):
    advanced_level: int

```

## Strict Mode & ValidationError

```python

from pydantic import BaseModel, ValidationError, ConfigDict
from typing import List

class Address(BaseModel):
    street: str
    city: str
    country: str

class User(BaseModel):

    model_config = ConfigDict(strict=True)

    id: int
    name: str
    email: str
    addresses: List[Address]
    age: int | None = None

```

```python
try:
    User(
        id = 123,
        name = "John Doe",
        email = "john.doe@example.com",
        addresses=[{"street": "123 Main St", "city": "Hometown", "country": "USA"}],
        age = 29,
    )
except ValidationError as e:
    print(e)

```

## 개별로 Strict 적용하기

```python
from pydantic import Strict, Field
from typing import List, Annotated, Optional

# 1. Typing hint
age: int | None = None # age: Optional[int] = None

# 2. Field 이용
age: int = Field(None, strict=True) # Optional

age: int = Field(strict=True) # Mandatory

# 3. Annotated 이용
age: Annotated[int, Strict()] = None # Optional

age: Annotated[int, Strict()] # Mandatory

```

## Field 사용법

- `lt`: less than
- `gt`: greater than
- `le`: less than or equal to
- `ge`: greater than or equal to
- `min_length`: 문자열 최소 길이
- `max_length`: 문자열 최대 길이


```python
password: str = Field(min_length=8)
age: Optional[int] = Field(None, ge=0, le=120)
short: str = Field(min_length=3)
long: str = Field(max_length=10)

```
## Pydantic에서 기본 제공 타입

`EmailStr`

```python

email: EmailStr # 자체적으로 email 형식의 ValidationError 확인

email: EmailStr = Field(None, max_length=40)

http_url: HttpUrl # http or https 프로토콜만 가능하다

file_url: FileUrl # file://로 시작하는 형식이여 야한다 

calss UrlModel(BaseModel):

    http_url: HttpUrl
    file_url: FileUrl

UrlModel(http_url = "https://www.example.com", 
         file_url = "file://path/to/file.txt)

```

## Validation Decoration 사용해보기 

`field_decorator`: 개별 인자에 대한 형식 체크
`model_decorator`: 여러 인자에 통합 형식 체크

```python
from pydantic import BaseModel, ValidationError
from pydantic import field_validator, model_validator

class User(BaseModel):
    username: str
    password: str
    confirm_password: str

    @field_validator('username') # 인자랑 이름 동일해야 한다 
    def username_must_not_be_empty(cls, value: str):
        if not value.strip():
            raise ValueError("Username must not be empty")
        return value

    @field_validator('password') # 인자랑 이름 동일해야 한다
    def password_must_be_strong(cls, value: str):
        if len(value) < 8:
            raise ValueError("Password must be a least 8 characters long")
        if not any(char.isdigit() for char in value):
            raise ValueError("Password must contain at least one digit")
        if not any(char.isalpha() for char in value):
            raise ValueError("Password must contain at least one letter")
        return value

    @model_validator(mode="after")
    def check_passwords_match(cls, values):
        password = values.password
        confirm_password = values.confirm_password
        if password != confirm_password:
            raise ValueError("Password do not match")
        return value

    
```

## Path, Query Parameter 검증

```python

from fastapi import FastAPI
from pydantic import Path, Query

app = FastAPI()
@app.put("/item_json/{item_id}")
async def update_item_json(
    item_id: int = Path(..., gt=0), # Annotated[int, Path(gt=0)]
    q1: str = Query(None, max_length=50), # Annotated[str, Query(max_length=50)] = None
    q2: str = Query(None, min_length=10), # Annotated[str, Query(min_length=10)] = None
)

```
