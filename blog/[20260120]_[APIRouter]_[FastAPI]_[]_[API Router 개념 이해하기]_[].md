# APIRouter 이해

> <ins/>늘어나는 업무 로직을 하나의 파일(main.py)에만 기재할수가 없음. 독립 가능한 업무레벨로, 구현 팀별로, 서로 다른 API version에 따라 별도의 구현 파일로 분할하여 구현 필요</ins>

> <ins/>FastAPI는 이를 위해 APIRouter를 제공하여 보다 조직화 되고, 쉬운 유지보수와 확장성 있는 코드 구현 가능</ins>

## main.py 

- main.py와 동일한 다이렉토리에 routes 폴더 생성
- routes 폴더 안에 main 기능을 item.py, user.py로 쪼개서 관리한다 

```bash
project_root/
├── main.py
└── routes/
    ├── item.py
    └── user.py

```

```python

from fastapi import FastAPI
from routes import item, user

app = FastAPI()

app.include_router(item.router)
app.include_router(user.router)

```

##  item.py

```python
from fastapi import APIRouter

router = APIRouter(prefix="/item", tags=["item"])

@router.get("/{item_id}")
async def read_item(item_id: int):
    return {"item_id", item_id}

```

## user.py

```python
from fastapi import APIRouter

router = APIRouter(prefix="/user", tags=['user'])

@router.get("/")
async def read_users():
    return None

```

