# 1. 파이썬 3.10 환경 사용
FROM python:3.10-slim

# 2. 필수 시스템 패키지 설치 (DeepFace 실행에 필요)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /code

# 4. 라이브러리 목록 복사 및 설치
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /code/requirements.txt

# 5. 모든 코드 복사
COPY . .

# 6. FastAPI 서버 실행 (포트 7860은 허깅페이스 기본 포트)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]