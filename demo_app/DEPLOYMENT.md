# 데모 배포 준비

## 결론

현재 구조는 React 빌드 결과와 FastAPI API를 한 프로세스가 제공하는 단일 컨테이너 서비스임. 1차 배포 플랫폼은 Google Cloud Run을 권장함.

- CPU 전용 PPO 추론이며 GPU가 필요하지 않음
- 컨테이너 이미지와 환경변수 기반 실행을 직접 지원함
- 서울 리전 `asia-northeast3` 사용 가능함
- 트래픽이 없을 때 0개 인스턴스로 축소 가능함
- Secret Manager의 값을 환경변수로 주입 가능함

Cloud Run은 `0.0.0.0`과 지정된 `PORT` 수신을 요구하며, 컨테이너 파일 시스템은 메모리 기반이고 인스턴스 종료 시 유지되지 않음. 현재 앱은 이 계약에 맞게 `HOST`와 `PORT`를 읽음.

공식 문서:

- [Cloud Run 컨테이너 계약](https://docs.cloud.google.com/run/docs/container-contract)
- [Cloud Run 메모리 설정](https://docs.cloud.google.com/run/docs/configuring/services/memory-limits)
- [Cloud Run 컨테이너 배포](https://docs.cloud.google.com/run/docs/deploying)
- [Cloud Run Secret Manager 연동](https://docs.cloud.google.com/run/docs/configuring/services/secrets)
- [Cloud Run health check](https://docs.cloud.google.com/run/docs/configuring/healthchecks)

## 포함 자산

| 자산 | 크기 | 이미지 포함 | 용도 |
| --- | ---: | --- | --- |
| PPO `best_model.zip` | 10.8MB | 포함 | 50차원 action 추론 |
| `encoder_dim16.pth` | 17.9MB | 포함 | Upstage 4,096차원 압축 |
| 시장 임베딩 JSONL | 14.9MB | 포함 | 과거 시장 뉴스 입력 |
| 종목 임베딩 JSONL | 97.0MB | 포함 | 과거 종목 뉴스 입력 |

필수 자산 합계는 약 140.6MB임. 네 파일 모두 현재 Git 추적 상태이며 Dockerfile은 다른 체크포인트와 학습 결과를 제외함.

## 환경변수

| 이름 | 필수 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `UPSTAGE_API_KEY` | 최신 뉴스 사용 시 필수 | 없음 | Secret Manager로 주입 |
| `UPSTAGE_BASE_URL` | 선택 | `https://api.upstage.ai/v1` | Upstage OpenAI 호환 API |
| `UPSTAGE_EMBEDDING_MODEL` | 선택 | `embedding-passage` | 4,096차원 임베딩 모델 |
| `HOST` | 컨테이너에서 필수 | 로컬 `127.0.0.1` | Dockerfile은 `0.0.0.0` 지정 |
| `PORT` | 선택 | 로컬 `8000` | Dockerfile과 Cloud Run은 `8080` 사용 |

루트 `.env.example`에는 키 이름과 비밀이 아닌 기본값만 포함함. 실제 `.env`는 이미지와 Git에서 제외됨.

## 로컬 컨테이너

Apple Silicon에서 Cloud Run용 이미지를 확인하려면 Linux AMD64로 빌드함.

```bash
docker build --platform linux/amd64 -t rlfinanceagent-demo:local .
docker run --rm --platform linux/amd64 \
  --env-file .env \
  -p 8080:8080 \
  rlfinanceagent-demo:local
```

확인:

```bash
curl --fail http://127.0.0.1:8080/api/live
curl --fail http://127.0.0.1:8080/api/health
open http://127.0.0.1:8080
```

`/api/live`는 프로세스 생존 여부만 확인함. `/api/health`는 PPO와 Autoencoder를 실제 로드하고 차원 계약을 확인하는 readiness endpoint임.

## Cloud Run 권장 설정

초기값:

- Region: `asia-northeast3`
- CPU: 1 vCPU
- Memory: 2GiB
- Concurrency: 1
- Request timeout: 300초
- Minimum instances: 0
- Maximum instances: 1
- Port: 8080
- Startup probe: `/api/health`
- Liveness probe: `/api/live`

로컬 네이티브 Python 최신 추론 후 RSS는 약 245MB였음. 배포 대상과 같은 Linux AMD64 컨테이너를 Apple Silicon에서 에뮬레이션한 측정은 cold model load 14.5초, peak RSS 597MiB, 최신 모드 83.6초, 최신 추론 후 컨테이너 메모리 약 911MiB였음. 에뮬레이션 오버헤드가 포함되지만 Cloud Run 파일 쓰기도 메모리를 사용하므로 2GiB로 시작하는 편이 안전함. 동시 추론은 앱 내부 lock으로 직렬화되므로 concurrency 1이 맞음.

최종 Linux AMD64 이미지 크기는 약 500.3MB임. CPU 전용 PyTorch wheel을 사용하며 CUDA 패키지는 포함하지 않음.

예시 배포 흐름이며 실제 실행은 사용자 승인 후 진행함.

```bash
PROJECT_ID="your-project-id"
REGION="asia-northeast3"
REPOSITORY="rlfinanceagent"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/demo:latest"

gcloud artifacts repositories create "${REPOSITORY}" \
  --repository-format=docker \
  --location="${REGION}"

gcloud builds submit --tag "${IMAGE}" .

gcloud run deploy rlfinanceagent-demo \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --cpu 1 \
  --memory 2Gi \
  --concurrency 1 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 1 \
  --port 8080 \
  --set-env-vars UPSTAGE_BASE_URL=https://api.upstage.ai/v1,UPSTAGE_EMBEDDING_MODEL=embedding-passage \
  --set-secrets UPSTAGE_API_KEY=upstage-api-key:1
```

`upstage-api-key`는 먼저 Secret Manager에 만들고 서비스 계정에 Secret Accessor 권한을 부여해야 함. 키 원문을 명령행, 이미지, Git에 넣지 않음.

## 외부 요청과 runtime 캐시

- Yahoo Finance, 네이버 검색, Upstage API는 일반 HTTPS outbound 요청임. Cloud Run은 인터넷 outbound 연결을 지원함.
- 네이버 검색은 공식 데이터 API가 아닌 HTML 검색 결과를 사용하므로 HTTP 403/429, selector 변경, 공유 egress IP 차단 가능성이 있음. 로컬 샘플 검증 중에도 연속 요청 후 HTTP 403이 발생했으며, 코드는 우회하지 않고 기존 최신 임베딩으로 fallback함.
- `demo_app/backend/runtime/`이 비어 있어도 모델과 과거 임베딩은 이미지에 있으므로 샘플 모드와 뉴스 없는 최신 모드는 안전함.
- runtime 손실 시 최신 뉴스 CSV·Upstage 임베딩·scaler를 다시 생성하므로 결과 손상은 없지만 지연, API 비용, 네이버 재수집 실패 가능성이 생김.
- 최대 인스턴스를 1로 제한하면 중복 크롤링을 줄일 수 있으나 재시작·재배포 시 캐시는 사라짐. 안정 운영이 필요하면 runtime 캐시를 Cloud Storage 같은 영속 저장소로 이전해야 함.

## 남은 배포 전 확인

1. Cloud Run 프로젝트, 결제, Artifact Registry와 Secret Manager 준비
2. 컨테이너 이미지의 실제 압축 크기와 cold start 확인
3. Cloud Run에서 Yahoo 50종목 다운로드 성공 여부 확인
4. Cloud Run egress IP에서 네이버 성공 또는 기존 임베딩 fallback 확인
5. `/api/health` startup probe와 최신 모드 300초 timeout 확인
6. 공개 URL로 둘 경우 API 호출량·Upstage 비용 제한 추가 검토
