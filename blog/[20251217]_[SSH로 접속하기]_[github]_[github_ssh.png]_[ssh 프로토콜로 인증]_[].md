# SSH를 통한 GitHub 연결

SSH 프로토콜을 사용하여 원격 서버 및 서비스에 연결하고 인증할 수 있다. SSH 키를 사용하면 방문할떄 마다 사용자 이름과 personal access token을 제공하지 않고 Github에 연결할 수 있다. 

- [x] _**공개키** 암호화 방식 활용_  
- [x] _username과 토큰 사용할 필요 없음_  
- [x] _로컬 컴퓨터 자체에 키 저장_  

## SSH 키 등록하기

- 계정의 `Settings` - `SSH and GPG keys`
- SSH 관련 GitHub 문서 [Click](https://docs.github.com/ko/authentication/connecting-to-github-with-ssh)

- 이미 SSH 키가 등록되어 있을 경우 

![img](/develop_blog/img/ssh_key.JPG)

### 1. SSH 키 존재 여부 확인

- 터미널(윈도우의 경우 Bash Shell)에서 `~/.ssh`로 이동

```bash
cd ~/.ssh
```
- `id_rsa.pub`, `id_ecdsa.pub`, `id_ed25519.pub` 파일 중 하나 존재 여부 확인

```bash
ls
```

### 2. 로컬 PC 내 SSH 키 생성 

- 터미널(윈도의 경우 Bash Shell)에서 키 생성

```bash
ssh-keygen -t ed25519 -C "(이메일 주소)"
```

- 1번의 과정으로 로컬 PC 내 키 생성 확인

### 3. Github에 키 등록

- 공개키 열람하여 복사

```bash
cat ~/.ssh/id_ed25519.pub
```

- `New SSH Key` 클릭하여 키 이름과 함께 등록

### 4. SSH로 사용해보기

원격을 SSH 주소로 변경한 뒤 테스트

### 5. SSH 활용하기 (Clone & Remote)

SSH 키 등록을 마쳤다면, 이제 HTTPS 대신 SSH 주소를 사용하여 깃허브 저장소를 관리할 수 있다. 

- **`git clone`**

```bash
git clone git@github.com:username/repository-name.git
```

- **`원격 주소를 SSH로 업데이트`**

```bash
git remote set-url origin git&github.com:username/repository-name.git
```