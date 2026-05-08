from __future__ import annotations  # (선택) 파이썬 타입 힌트 호환성을 조금 더 좋게 해줘요

import os  # 파일 삭제 같은 OS 기능을 쓰기 위해 가져와요
import tempfile  # 업로드 파일을 임시 파일로 저장하기 위해 가져와요
from typing import Any, Dict, List, Optional  # 타입(자료형) 힌트를 쓰기 위해 가져와요

import requests  # URL로 이미지를 다운로드하기 위해 가져와요
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # 파일 업로드/폼 데이터 처리를 위해 가져와요
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # 응답 데이터 검증(스키마)을 위해 가져와요

from deepface import DeepFace  # 얼굴 임베딩(특징 벡터)을 추출하기 위해 DeepFace를 가져와요
from pinecone import Pinecone  # Pinecone 벡터 DB에 연결하기 위한 클라이언트를 가져와요

# -----------------------------
# 1) FastAPI 앱 생성
# -----------------------------

app = FastAPI(  # FastAPI 애플리케이션(서버)을 만들어요
    title="Face Vector API",  # API 문서에 표시될 제목이에요
    version="1.0.0",  # API 버전이에요
)

# 17번 줄 근처에 있는 app 생성 코드 바로 밑에 넣어주세요!
app = FastAPI(
    title="Face Vector API",
    version="1.0.0",
)

# ---- 여기서부터 복사해서 붙여넣기 ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 모든 사이트의 접속을 허용 (매우 중요!)
    allow_credentials=False,       # origins가 "*"일 때는 반드시 False여야 함
    allow_methods=["*"],           # 모든 통신 방식(GET, POST 등) 허용
    allow_headers=["*"],           # 모든 데이터 헤더 허용
)
# ---- 여기까지 ----

# -----------------------------
# 2) Pinecone 설정 및 연결
# -----------------------------

PINECONE_API_KEY = "pcsk_6U7zkn_Rp4SYP4hkZYYHxzuAn3jKqQ3vpLhdsGCdocvT7o4pZdT6tvejstZFRXRj7zdVzn"  # (임시) Pinecone API 키를 여기에 적어둬요 (실사용 시 실제 키로 교체)
PINECONE_INDEX_NAME = "face-db"  # 연결할 Pinecone 인덱스 이름이에요
VECTOR_DIMENSION = 512  # 목표 임베딩 차원 수예요 (Facenet512는 512차원)
SIMILARITY_THRESHOLD = 0.5  # 코사인 유사도 0.9(90%) 이상만 “일치”로 판단할 거예요

pc = Pinecone(api_key=PINECONE_API_KEY)  # Pinecone 클라이언트를 생성해요


def get_index():  # 인덱스 객체를 안전하게 얻어오는 함수예요
    try:  # 아래 코드에서 에러가 날 수 있으니 예외 처리를 시작해요
        return pc.Index(PINECONE_INDEX_NAME)  # 'face-db' 인덱스에 연결한 객체를 반환해요
    except Exception as e:  # 인덱스 연결에 실패하면(예: 인덱스가 없거나 키가 잘못됨) 여기로 와요
        raise HTTPException(  # FastAPI에서 500 에러로 응답을 내려줘요
            status_code=500,  # 서버 내부 오류라는 뜻이에요
            detail={  # 에러 내용을 JSON 형태로 자세히 알려줘요
                "message": "Pinecone 인덱스 연결에 실패했어요. API 키/인덱스 이름을 확인해 주세요.",
                "index_name": PINECONE_INDEX_NAME,
                "hint": "Pinecone 콘솔에서 'face-db' 인덱스가 존재하는지 확인하세요. (dimension=512, metric=cosine 권장)",
                "error": str(e),
            },
        )


# -----------------------------
# 3) DeepFace(Facenet) 모델 준비
# -----------------------------

# DeepFace에서 모델 이름에 따라 임베딩 차원이 달라질 수 있어요.
# - "Facenet"     : 보통 128차원
# - "Facenet512"  : 512차원 (우리가 원하는 목표)
DEEPFACE_MODEL_NAME = "ArcFace"  # 512차원을 확실히 받기 위해 Facenet512로 강제해요


def extract_facenet_embedding(image_path: str) -> List[float]:  # 이미지 파일에서 512차원 임베딩을 뽑는 함수예요
    try:  # 얼굴이 없거나 파일이 잘못되면 에러가 날 수 있어서 예외 처리를 해요
        reps = DeepFace.represent(  # DeepFace로 얼굴 임베딩을 추출해요
            img_path=image_path,  # 분석할 이미지 파일 경로예요
            model_name=DEEPFACE_MODEL_NAME,  # Facenet512를 강제해서 512차원을 얻도록 해요
            detector_backend="retinaface",
            align=True,
            enforce_detection=True,  # 얼굴이 없으면 에러로 처리해서 “등록/스캔 실패”를 명확히 해요
        )
    except ValueError as ve: # 🚨 딱 여기! 얼굴을 못 찾았을 때 발생하는 에러를 낚아채요!
        raise HTTPException(
            status_code=400,
            detail={"message": "사진에서 얼굴을 찾을 수 없어요! 😢 정면이 잘 보이는 사진으로 다시 시도해 주세요.", "error": str(ve)}
        )
    except Exception as e:  # DeepFace 단계에서 실패하면 여기로 와요
        raise HTTPException(  # 400(클라이언트 입력 문제)로 안내하는 게 일반적으로 친절해요
            status_code=400,  # 잘못된 입력(얼굴 없음/이미지 손상 등)으로 처리해요
            detail={"message": "이미지에서 얼굴 임베딩 추출에 실패했어요. 얼굴이 선명하게 나온 이미지인지 확인해 주세요.", "error": str(e)},
        )

    if not reps:  # represent 결과가 비어있다면(매우 드문 케이스) 처리해요
        raise HTTPException(  # 400 에러로 응답해요
            status_code=400,  # 입력 이미지 문제일 가능성이 높아요
            detail={"message": "임베딩을 만들지 못했어요. 다른 이미지로 다시 시도해 주세요."},
        )

    rep0 = reps[0]  # 여러 얼굴이 있어도 일단 첫 번째 얼굴 결과를 사용해요(간단 버전)
    embedding = rep0.get("embedding") if isinstance(rep0, dict) else None  # dict 형태에서 embedding을 꺼내요

    if not isinstance(embedding, list):  # embedding이 리스트가 아니면 오류로 처리해요
        raise HTTPException(  # 500 에러로 처리해요(서버/라이브러리 응답 형태가 예상과 다름)
            status_code=500,
            detail={"message": "DeepFace 임베딩 결과 형식이 예상과 달라서 처리할 수 없어요."},
        )

    if len(embedding) != VECTOR_DIMENSION:  # 512차원이 아니면 여기서 원인을 분명하게 알려줘요
        # 자주 발생하는 케이스: "Facenet"이 실제로 적용되어 128차원이 나오는 경우예요
        if len(embedding) == 128:  # 128차원이라면 Facenet(128)로 추출됐을 가능성이 커요
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "임베딩이 128차원으로 나왔어요. 현재 DeepFace가 Facenet512(512차원) 대신 Facenet(128차원) 또는 다른 모델로 동작 중일 수 있어요.",
                    "expected_dimension": VECTOR_DIMENSION,
                    "actual_dimension": len(embedding),
                    "configured_model_name": DEEPFACE_MODEL_NAME,
                    "hint": "DeepFace에서 512차원은 보통 model_name='Facenet512'입니다. 그래도 128이 나오면 DeepFace 버전/모델 지원 여부를 확인하거나 Pinecone 인덱스를 128차원으로 맞추는 방식을 고려해야 해요.",
                },
            )

        raise HTTPException(  # 그 외 차원은 환경/모델 문제로 보고 500으로 처리해요
            status_code=500,
            detail={
                "message": "임베딩 차원이 예상과 달라요. 모델/설정을 확인해 주세요.",
                "expected_dimension": VECTOR_DIMENSION,
                "actual_dimension": len(embedding),
                "configured_model_name": DEEPFACE_MODEL_NAME,
            },
        )

    return [float(x) for x in embedding]  # Pinecone에 넣기 좋게 float 리스트로 보정해서 반환해요


# -----------------------------
# 4) 응답 모델 정의(Pydantic)
# -----------------------------


class RegisterVictimResponse(BaseModel):  # /register_victim 응답 형태를 정의해요
    user_id: str  # 등록된 피해자(사용자) ID예요
    dimension: int  # 저장된 임베딩 차원 수예요(512)


class ScanImageResponse(BaseModel):  # /scan_image 응답 형태를 정의해요
    threshold: float  # 사용한 유사도 임계값(0.9)이에요
    matched: bool  # 임계값 이상 매칭이 있었는지 여부예요
    matches: List[Dict[str, Any]]  # 임계값 이상인 매칭 결과 목록이에요


# -----------------------------
# 5) 엔드포인트 구현
# -----------------------------


@app.post("/register_victim", response_model=RegisterVictimResponse)  # POST /register_victim 엔드포인트를 만들어요
async def register_victim(  # 업로드된 이미지로 피해자(등록 대상) 임베딩을 추출해서 저장해요
    user_id: str = Form(...),  # 파일 업로드와 같이 보내는 텍스트 값은 Form으로 받아요
    image: UploadFile = File(...),  # 이미지 파일은 UploadFile로 받아요
) -> RegisterVictimResponse:
    index = get_index()  # Pinecone 인덱스를 가져와요

    suffix = os.path.splitext(image.filename or "")[1]  # 원본 파일 확장자를 최대한 유지해요(없으면 빈 문자열)
    tmp_path = None  # 임시 파일 경로를 나중에 삭제하기 위해 저장할 거예요

    try:  # 임시 파일 저장/분석 과정에서 문제가 생길 수 있어서 try를 써요
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:  # delete=False로 해야 Windows에서 재열기 이슈가 덜해요
            tmp_path = tmp.name  # 만들어진 임시 파일 경로를 기록해요
            content = await image.read()  # 업로드된 파일 내용을 메모리로 읽어와요(큰 파일이면 제한이 필요할 수 있어요)
            tmp.write(content)  # 임시 파일에 그대로 저장해요

        embedding = extract_facenet_embedding(tmp_path)  # 임시 파일 경로를 DeepFace에 넘겨 임베딩을 추출해요

        vectors = [  # Pinecone upsert 형식에 맞춰 리스트로 만들어요
            {  # 벡터 1개를 표현하는 dict예요
                "id": user_id,  # 벡터의 고유 ID는 user_id로 사용해요(동일 ID면 갱신돼요)
                "values": embedding,  # 512차원 임베딩 값을 저장해요
                "metadata": {"user_id": user_id, "model": DEEPFACE_MODEL_NAME},  # 메타데이터로 user_id/모델명을 같이 저장해요
            }
        ]

        index.upsert(vectors=vectors)  # Pinecone에 벡터를 저장(또는 업데이트)해요

    except HTTPException:  # 우리가 이미 만든 HTTPException은 그대로 다시 던져요
        raise
    except Exception as e:  # 그 외 예기치 못한 에러는 500으로 처리해요
        raise HTTPException(  # 서버 내부 오류로 응답해요
            status_code=500,  # 내부 오류예요
            detail={"message": "피해자 등록 처리 중 오류가 발생했어요.", "error": str(e)},
        )
    finally:  # 성공/실패와 상관없이 임시 파일은 삭제를 시도해요
        if tmp_path and os.path.exists(tmp_path):  # 임시 파일 경로가 있고 실제로 존재하면 실행해요
            try:  # 삭제 중에도 에러가 날 수 있어서 한 번 더 보호해요
                os.remove(tmp_path)  # 임시 파일을 삭제해요
            except Exception:  # 삭제 실패는 치명적이지 않아서 조용히 넘어가요
                pass

    return RegisterVictimResponse(  # 응답을 반환해요
        user_id=user_id,  # 등록된 user_id를 알려줘요
        dimension=VECTOR_DIMENSION,  # 512차원임을 알려줘요
    )


@app.post("/scan_image", response_model=ScanImageResponse)  # POST /scan_image 엔드포인트를 만들어요
async def scan_image(  # 의심 이미지에서 임베딩을 뽑아 Pinecone에서 피해자 매칭을 찾아요
    image: Optional[UploadFile] = File(None),  # 업로드된 이미지 파일(없을 수도 있어요)
    url: Optional[str] = Form(None),  # 이미지 URL(없을 수도 있어요)
) -> ScanImageResponse:
    index = get_index()  # Pinecone 인덱스를 가져와요

    tmp_path = None  # finally에서 삭제할 수 있도록 임시 경로를 저장해요

    try:  # 임시 파일 저장/분석 과정 에러를 처리하기 위해 try를 써요
        if image is None and (url is None or not url.strip()):  # 둘 다 없으면 요청 자체가 잘못된 거예요
            raise HTTPException(  # 400 에러로 응답해요
                status_code=400,  # 클라이언트 요청 오류예요
                detail={"message": "image 파일 또는 url 중 하나는 반드시 제공해야 해요."},
            )

        if url is not None and url.strip():  # url이 들어왔으면 URL 다운로드 방식으로 처리해요
            # URL에서 확장자를 추정해 임시 파일 이름에 반영해요(필수는 아니지만 도움이 돼요)
            suffix = os.path.splitext(url.split("?", 1)[0])[1]  # 쿼리스트링은 제거하고 확장자만 추출해요
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:  # Windows 호환을 위해 delete=False를 써요
                tmp_path = tmp.name  # 임시 파일 경로를 기록해요

                try:  # 다운로드 중 네트워크/HTTP 오류가 날 수 있어요
                    r = requests.get(url, timeout=15)  # URL에서 이미지를 다운로드해요(타임아웃을 줘요)
                    r.raise_for_status()  # 4xx/5xx면 예외를 발생시켜요
                except Exception as e:  # 다운로드 실패 시 여기로 와요
                    raise HTTPException(  # 400으로 처리해요(보통 URL/접근 문제)
                        status_code=400,
                        detail={"message": "url에서 이미지를 다운로드하지 못했어요.", "url": url, "error": str(e)},
                    )

                tmp.write(r.content)  # 다운로드한 바이트를 임시 파일에 저장해요

        else:  # url이 없고 image가 있으면 기존 업로드 방식으로 처리해요
            suffix = os.path.splitext((image.filename if image else "") or "")[1]  # 확장자를 유지해 임시 파일을 만들어요
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:  # Windows 호환을 위해 delete=False를 써요
                tmp_path = tmp.name  # 임시 파일 경로를 기록해요
                content = await image.read()  # 업로드 파일 내용을 읽어와요
                tmp.write(content)  # 임시 파일에 저장해요

        embedding = extract_facenet_embedding(tmp_path)  # DeepFace로 512차원 임베딩을 뽑아요

        result = index.query(  # Pinecone에서 코사인 유사도로 검색해요(인덱스 metric=cosine 권장)
            vector=embedding,  # 검색 벡터예요
            top_k=10,  # 후보를 좀 넉넉히 받아온 뒤 임계값으로 걸러요
            include_metadata=True,  # user_id 같은 메타데이터도 같이 받기 위해 True로 해요
        )

        matches = result.get("matches") if isinstance(result, dict) else getattr(result, "matches", None)  # matches 목록을 꺼내요
        matches = matches or []  # None이면 빈 리스트로 바꿔서 아래 로직을 단순화해요

        filtered: List[Dict[str, Any]] = []  # 임계값 이상 매칭 결과만 모을 리스트예요
        for m in matches:  # 검색 결과를 하나씩 살펴봐요
            if isinstance(m, dict):  # dict 형태라면 키로 접근해요
                score = float(m.get("score") or 0.0)  # score가 없으면 0으로 처리해요
                mid = m.get("id")  # 매칭된 벡터 ID예요
                meta = m.get("metadata")  # 메타데이터예요
            else:  # 객체 형태라면 속성으로 접근해요
                score = float(getattr(m, "score", 0.0) or 0.0)  # score를 꺼내요
                mid = getattr(m, "id", None)  # id를 꺼내요
                meta = getattr(m, "metadata", None)  # metadata를 꺼내요

            if score >= SIMILARITY_THRESHOLD:  # 코사인 유사도가 0.9 이상이면 “일치”로 판단해요
                filtered.append(  # 응답에 넣기 좋은 형태로 정리해서 추가해요
                    {"id": mid, "score": score, "metadata": meta}  # 필요한 정보만 담아요
                )

        return ScanImageResponse(  # 최종 응답을 반환해요
            threshold=SIMILARITY_THRESHOLD,  # 사용한 임계값을 알려줘요
            matched=len(filtered) > 0,  # 하나라도 있으면 matched=True예요
            matches=filtered,  # 임계값 이상인 결과만 반환해요
        )

    except HTTPException:  # 이미 만든 HTTPException은 그대로 반환해요
        raise
    except Exception as e:  # 그 외 예기치 못한 에러는 500으로 처리해요
        raise HTTPException(  # 내부 오류로 응답해요
            status_code=500,  # 서버 내부 오류예요
            detail={"message": "이미지 스캔 처리 중 오류가 발생했어요.", "error": str(e)},
        )
    finally:  # 성공/실패와 상관없이 임시 파일은 삭제해요
        if tmp_path and os.path.exists(tmp_path):  # 임시 파일이 실제로 존재하면 실행해요
            try:  # 삭제 중 오류가 날 수 있어요
                os.remove(tmp_path)  # 임시 파일을 삭제해요
            except Exception:  # 삭제 실패는 치명적이지 않아서 넘어가요
                pass


# -----------------------------
# 6) 로컬 실행(옵션)
# -----------------------------

if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만 아래가 동작해요
    import uvicorn  # ASGI 서버인 uvicorn을 가져와요

    uvicorn.run(  # uvicorn으로 FastAPI 앱을 실행해요
        "main:app",  # main.py의 app 객체를 실행 대상으로 지정해요
        host="127.0.0.1",  # 로컬 PC에서만 접속되도록 localhost로 열어요
        port=8000,  # 기본 포트는 8000으로 열어요
        reload=True,  # 코드가 바뀌면 자동으로 재시작되게 해요(개발용)
    )
