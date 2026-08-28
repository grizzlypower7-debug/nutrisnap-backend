"""
NutriSnap backend — FastAPI сервер с поддержкой SQLite (для локальной
разработки) и PostgreSQL (для продакшена, например Neon.tech + Render).

Функционал:
- Хранит пользователей и их сканы. По умолчанию — в SQLite-файле рядом
  с сервером. Если задана переменная окружения DATABASE_URL — все данные
  идут в PostgreSQL (Neon и подобные), что переживает redeploy на Render
  даже без платного постоянного диска.
- POST /scan-meal — принимает фото (файл) или текст, анализирует блюдо
  через gpt-4o-mini (ProxyAPI) и возвращает КБЖУ.
- Лимит: ровно 2 бесплатных скана за скользящее окно в 24 часа (86400
  секунд от МОМЕНТА конкретного скана, а не от календарной полуночи).
  На 3-й скан за окно — 402 Payment Required с полем next_scan_in_seconds.
- Telegram Stars: /create-star-invoice + /confirm-payment активируют
  Premium на 30 дней (premium_until), который полностью снимает лимит.
- Если ИИ недоступен (нет ключа, сбой сети, невалидный ответ) — сервер
  никогда не падает и тихо откатывается на тестовый mock-результат.

Все временные метки везде трактуются как "наивный" UTC (без tzinfo) —
и в SQLite (TEXT в формате 'YYYY-MM-DD HH:MM:SS'), и в PostgreSQL
(TIMESTAMP WITHOUT TIME ZONE) — чтобы сравнения между Python/SQLite/
PostgreSQL всегда были на одной временной шкале.
"""

import base64
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Optional

import httpx
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # psycopg2-binary мог быть не установлен в чисто sqlite-окружении
    psycopg2 = None

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")  # например, строка подключения Neon.tech
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES and psycopg2 is None:
    raise RuntimeError(
        "DATABASE_URL задан, но psycopg2 не установлен. "
        "Добавьте psycopg2-binary в requirements.txt и переустановите зависимости."
    )

DB_PATH = os.getenv("DB_PATH", "nutrisnap.db")  # используется только для SQLite-фоллбэка

FREE_DAILY_LIMIT = 2  # <-- ровно 2 бесплатных скана за скользящее окно в 24 часа
LIMIT_WINDOW_SECONDS = 86400  # 24 часа

AI_MODEL = "gpt-4o-mini"
AI_SYSTEM_PROMPT = (
    "Ты — профессиональный нутрициолог. Оцени тарелку на фото или текст. "
    "Верни СТРОГО чистый JSON без маркдауна: "
    '{"meal": "Название блюда на русском", "calories": 450, "protein": 30, '
    '"fat": 12, "carbs": 45}'
)

api_key = os.getenv("OPENAI_API_KEY")
client = (
    OpenAI(
        api_key=api_key,
        base_url="https://api.proxyapi.ru/openai/v1",
    )
    if api_key
    else None
)

# --- Telegram Stars (оплата Premium) ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PREMIUM_PRICE_STARS = 150  # 150 XTR ≈ 300 ₽
PREMIUM_DURATION_SECONDS = 2592000  # 30 дней

app = FastAPI(title="NutriSnap API")

# Для разработки разрешаем все источники (Telegram WebView, localhost и т.д.)
# В проде сузьте allow_origins до вашего домена фронтенда.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Слой доступа к БД: единый интерфейс поверх SQLite и PostgreSQL.
#
# Бизнес-логика (лимиты, Stars, ИИ) ничего не знает о том, какая именно
# СУБД используется — она просто вызывает db_execute(conn, sql, params)
# и читает строки как словарь (row["colname"]), что одинаково работает
# и для sqlite3.Row, и для psycopg2 RealDictCursor.
#
# ВАЖНО (продакшен): здесь каждый запрос открывает новое соединение с БД
# и закрывает его в конце ("connect-per-request"). Для демо/MVP с Neon
# free tier этого достаточно, но под реальной нагрузкой лучше завести
# пул соединений (psycopg2.pool / SQLAlchemy) — открытие нового TCP+TLS
# соединения на КАЖДЫЙ HTTP-запрос заметно медленнее и может упереться
# в лимит одновременных соединений бесплатного плана Neon.
# ---------------------------------------------------------------------------

@contextmanager
def get_db():
    if USE_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def db_execute(conn, sql: str, params: tuple = ()):
    """
    Выполняет запрос и возвращает объект с .fetchone()/.fetchall(),
    одинаково для sqlite3.Connection и psycopg2-соединения.
    """
    if USE_POSTGRES:
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur
    return conn.execute(sql, params)


def init_db():
    with get_db() as conn:
        if USE_POSTGRES:
            db_execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    telegram_id TEXT UNIQUE NOT NULL,
                    daily_limit INTEGER NOT NULL DEFAULT 2,
                    is_premium BOOLEAN NOT NULL DEFAULT FALSE,
                    premium_until TIMESTAMP,
                    created_at TIMESTAMP NOT NULL DEFAULT (NOW() AT TIME ZONE 'utc')
                )
                """,
            )
            db_execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS scans (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    scan_date TEXT NOT NULL,
                    input_type TEXT NOT NULL,
                    meal_name TEXT,
                    calories INTEGER,
                    protein INTEGER,
                    fat INTEGER,
                    carbs INTEGER,
                    created_at TIMESTAMP NOT NULL DEFAULT (NOW() AT TIME ZONE 'utc')
                )
                """,
            )
            # Миграция на случай, если таблица users была создана до premium_until
            cur = db_execute(
                conn,
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'users'",
            )
            existing_columns = {row["column_name"] for row in cur.fetchall()}
            if "premium_until" not in existing_columns:
                db_execute(conn, "ALTER TABLE users ADD COLUMN premium_until TIMESTAMP")
        else:
            db_execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id TEXT UNIQUE NOT NULL,
                    daily_limit INTEGER NOT NULL DEFAULT 2,
                    is_premium INTEGER NOT NULL DEFAULT 0,
                    premium_until TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """,
            )
            db_execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    scan_date TEXT NOT NULL,
                    input_type TEXT NOT NULL,
                    meal_name TEXT,
                    calories INTEGER,
                    protein INTEGER,
                    fat INTEGER,
                    carbs INTEGER,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """,
            )
            existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
            if "premium_until" not in existing_columns:
                db_execute(conn, "ALTER TABLE users ADD COLUMN premium_until TEXT")


@app.on_event("startup")
def on_startup():
    init_db()
    print(f"[NutriSnap] БД: {'PostgreSQL (DATABASE_URL)' if USE_POSTGRES else f'SQLite ({DB_PATH})'}")


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def get_or_create_user(conn, telegram_id: str):
    ph = "%s" if USE_POSTGRES else "?"
    cur = db_execute(conn, f"SELECT * FROM users WHERE telegram_id = {ph}", (telegram_id,))
    user = cur.fetchone()
    if user is None:
        db_execute(
            conn,
            f"INSERT INTO users (telegram_id, daily_limit) VALUES ({ph}, {ph})",
            (telegram_id, FREE_DAILY_LIMIT),
        )
        cur = db_execute(conn, f"SELECT * FROM users WHERE telegram_id = {ph}", (telegram_id,))
        user = cur.fetchone()
    return user


def _as_naive_utc_datetime(value) -> Optional[datetime]:
    """
    Приводит значение времени из БД (строка из SQLite или datetime из
    PostgreSQL) к единому "наивному" datetime в UTC для сравнений в Python.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    return None


def _format_dt(value: Optional[datetime]) -> Optional[str]:
    """Единый строковый формат даты для JSON-ответов API, независимо от СУБД."""
    if value is None:
        return None
    return value.strftime("%Y-%m-%d %H:%M:%S")


def is_premium_active(user_row) -> bool:
    """True, если premium_until в будущем — тогда 24-часовой лимит не начисляется."""
    premium_dt = _as_naive_utc_datetime(user_row["premium_until"])
    if premium_dt is None:
        return False
    return premium_dt > datetime.utcnow()


def count_scans_last_24h(conn, user_id: int) -> int:
    """Сколько сканов пользователь сделал за последние 86400 секунд (скользящее окно)."""
    if USE_POSTGRES:
        cur = db_execute(
            conn,
            """
            SELECT COUNT(*) AS cnt FROM scans
            WHERE user_id = %s AND created_at >= (NOW() AT TIME ZONE 'utc') - INTERVAL '1 day'
            """,
            (user_id,),
        )
    else:
        cur = db_execute(
            conn,
            """
            SELECT COUNT(*) AS cnt FROM scans
            WHERE user_id = ? AND created_at >= datetime('now', '-1 day')
            """,
            (user_id,),
        )
    return cur.fetchone()["cnt"]


def seconds_until_next_scan(conn, user_id: int) -> int:
    """
    Через сколько секунд освободится следующий бесплатный скан —
    то есть когда самому старому скану за последние 24 часа "стукнет" 24 часа.
    Возвращает 0, если сканов за окно нет (лимит не исчерпан).
    Расчёт делается в Python (единообразно для SQLite и PostgreSQL), а не
    в SQL, чтобы не дублировать арифметику дат на двух диалектах.
    """
    if USE_POSTGRES:
        cur = db_execute(
            conn,
            """
            SELECT MIN(created_at) AS oldest FROM scans
            WHERE user_id = %s AND created_at >= (NOW() AT TIME ZONE 'utc') - INTERVAL '1 day'
            """,
            (user_id,),
        )
    else:
        cur = db_execute(
            conn,
            """
            SELECT MIN(created_at) AS oldest FROM scans
            WHERE user_id = ? AND created_at >= datetime('now', '-1 day')
            """,
            (user_id,),
        )
    row = cur.fetchone()
    oldest = _as_naive_utc_datetime(row["oldest"]) if row else None
    if oldest is None:
        return 0

    remaining = (oldest + timedelta(seconds=LIMIT_WINDOW_SECONDS) - datetime.utcnow()).total_seconds()
    return max(int(round(remaining)), 0)


def mock_analyze_meal() -> dict:
    """
    Заглушка (mock) анализа еды.
    Используется как fallback, если ИИ недоступен или вернул невалидный ответ —
    сервер никогда не должен падать из-за проблем с внешним API.
    """
    return {
        "meal": "Куриное филе с рисом",
        "calories": 450,
        "protein": 40,
        "fat": 6,
        "carbs": 50,
    }


def _parse_ai_json(raw_content: str) -> dict:
    """
    Парсит ответ модели в dict и валидирует обязательные поля.
    Бросает исключение при любой проблеме — вызывающий код ловит его
    и откатывается на mock_analyze_meal().
    """
    content = raw_content.strip()

    # На случай, если модель всё же обернула JSON в ```...``` markdown-блок
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
        content = content.strip()

    data = json.loads(content)

    result = {
        "meal": str(data["meal"]),
        "calories": int(round(float(data["calories"]))),
        "protein": int(round(float(data["protein"]))),
        "fat": int(round(float(data["fat"]))),
        "carbs": int(round(float(data["carbs"]))),
    }
    return result


def analyze_meal_text(text: str) -> dict:
    """Анализ текстового описания еды через gpt-4o-mini (ProxyAPI)."""
    if client is None:
        return mock_analyze_meal()

    try:
        response = client.chat.completions.create(
            model=AI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        raw_content = response.choices[0].message.content
        return _parse_ai_json(raw_content)
    except Exception as exc:  # сеть, лимиты API, невалидный JSON и т.д.
        print(f"[NutriSnap] Ошибка анализа текста через ИИ, использую mock: {exc}")
        return mock_analyze_meal()


def analyze_meal_photo(image_bytes: bytes, content_type: str) -> dict:
    """Анализ фото тарелки через Vision API gpt-4o-mini (ProxyAPI)."""
    if client is None:
        return mock_analyze_meal()

    try:
        mime = content_type if content_type else "image/jpeg"
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime};base64,{b64_image}"

        response = client.chat.completions.create(
            model=AI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Оцени калорийность и БЖУ блюда на этом фото.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "low"},
                        },
                    ],
                },
            ],
        )
        raw_content = response.choices[0].message.content
        return _parse_ai_json(raw_content)
    except Exception as exc:  # сеть, лимиты API, невалидный JSON и т.д.
        print(f"[NutriSnap] Ошибка анализа фото через ИИ, использую mock: {exc}")
        return mock_analyze_meal()


# ---------------------------------------------------------------------------
# Схемы ответов
# ---------------------------------------------------------------------------

class ScanResult(BaseModel):
    meal: str
    calories: int
    protein: int
    fat: int
    carbs: int
    scans_used_today: int
    scans_left_today: int
    next_scan_in_seconds: int
    is_premium: bool
    premium_until: Optional[str] = None


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

@app.post("/scan-meal", response_model=ScanResult)
async def scan_meal(
    telegram_id: str = Form(..., description="Telegram user id из initData"),
    text: Optional[str] = Form(None, description="Текстовое описание еды (голос -> текст)"),
    photo: Optional[UploadFile] = None,
):
    if not text and not photo:
        raise HTTPException(status_code=400, detail="Send either 'photo' or 'text'.")

    with get_db() as conn:
        user = get_or_create_user(conn, telegram_id)
        premium_active = is_premium_active(user)

        # Премиум (активный premium_until) полностью снимает 24-часовой лимит
        if not premium_active:
            used_last_24h = count_scans_last_24h(conn, user["id"])
            if used_last_24h >= FREE_DAILY_LIMIT:
                next_in = seconds_until_next_scan(conn, user["id"])
                raise HTTPException(
                    status_code=402,
                    detail={
                        "message": "Limit reached. Upgrade to Premium",
                        "next_scan_in_seconds": next_in,
                    },
                )

        # --- Реальный ИИ-анализ (gpt-4o-mini через ProxyAPI), с fallback на mock ---
        if photo is not None:
            image_bytes = await photo.read()
            result = analyze_meal_photo(image_bytes, photo.content_type)
        else:
            result = analyze_meal_text(text)

        input_type = "photo" if photo is not None else "text"
        today = date.today().isoformat()
        ph = "%s" if USE_POSTGRES else "?"

        db_execute(
            conn,
            f"""
            INSERT INTO scans
                (user_id, scan_date, input_type, meal_name, calories, protein, fat, carbs)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
            """,
            (
                user["id"],
                today,
                input_type,
                result["meal"],
                result["calories"],
                result["protein"],
                result["fat"],
                result["carbs"],
            ),
        )

        used_after = count_scans_last_24h(conn, user["id"])
        left = -1 if premium_active else max(FREE_DAILY_LIMIT - used_after, 0)
        next_in = 0 if premium_active or left > 0 else seconds_until_next_scan(conn, user["id"])

        return ScanResult(
            **result,
            scans_used_today=used_after,
            scans_left_today=left,
            next_scan_in_seconds=next_in,
            is_premium=premium_active,
            premium_until=_format_dt(_as_naive_utc_datetime(user["premium_until"])) if premium_active else None,
        )


@app.get("/scan-meal/status")
def scan_status(telegram_id: str):
    """Узнать, сколько сканов осталось за последние 24 часа, не тратя лимит."""
    with get_db() as conn:
        user = get_or_create_user(conn, telegram_id)
        premium_active = is_premium_active(user)
        used = count_scans_last_24h(conn, user["id"])
        left = -1 if premium_active else max(FREE_DAILY_LIMIT - used, 0)
        next_in = 0 if premium_active or left > 0 else seconds_until_next_scan(conn, user["id"])
        return {
            "telegram_id": telegram_id,
            "is_premium": premium_active,
            "premium_until": _format_dt(_as_naive_utc_datetime(user["premium_until"])) if premium_active else None,
            "scans_used_today": used,
            "scans_left_today": left,
            "daily_limit": FREE_DAILY_LIMIT,
            "next_scan_in_seconds": next_in,
        }


@app.post("/create-star-invoice")
async def create_star_invoice(telegram_id: str = Form(...)):
    """
    Создаёт ссылку на счёт Telegram Stars через Bot API (createInvoiceLink).
    Фронтенд открывает эту ссылку через Telegram.WebApp.openInvoice(...).
    """
    if not BOT_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="TELEGRAM_BOT_TOKEN не настроен на сервере",
        )

    payload = {
        "title": "NutriSnap Premium (1 месяц)",
        "description": "Безлимитные сканы еды по фото и ИИ-анализ БЖУ на 30 дней",
        "payload": f"premium_30_{telegram_id}",
        "currency": "XTR",
        "prices": [{"label": "1 месяц Premium", "amount": PREMIUM_PRICE_STARS}],
        # Для оплаты Telegram Stars (валюта XTR) provider_token обязан быть пустой строкой —
        # так требует Bot API, иначе createInvoiceLink вернёт ошибку.
        "provider_token": "",
    }

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/createInvoiceLink"
    try:
        async with httpx.AsyncClient(timeout=15) as http_client:
            response = await http_client.post(url, json=payload)
        data = response.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Telegram API недоступен: {exc}")

    if not data.get("ok"):
        raise HTTPException(
            status_code=502,
            detail=data.get("description", "Не удалось создать ссылку на оплату"),
        )

    return {"invoice_link": data["result"]}


@app.post("/confirm-payment")
def confirm_payment(telegram_id: str = Form(...)):
    """
    Активирует Premium на 30 дней для пользователя.

    ВАЖНО (безопасность): в этом MVP эндпоинт вызывается напрямую с фронтенда
    после колбэка openInvoice со статусом 'paid', без дополнительной проверки
    на сервере. Это удобно для быстрого демо, но в проде это дыра: любой
    человек может вызвать POST /confirm-payment с произвольным telegram_id
    и получить Premium бесплатно, без реальной оплаты.
    Для продакшена такую активацию должен делать бот через вебхук на апдейт
    successful_payment (Telegram сам присылает его после реальной оплаты),
    а не клиент, который присылает "доверьте мне, я оплатил".
    """
    with get_db() as conn:
        user = get_or_create_user(conn, telegram_id)
        new_premium_until = datetime.utcnow() + timedelta(seconds=PREMIUM_DURATION_SECONDS)
        ph = "%s" if USE_POSTGRES else "?"

        db_execute(
            conn,
            f"UPDATE users SET premium_until = {ph} WHERE id = {ph}",
            (
                new_premium_until if USE_POSTGRES else new_premium_until.strftime("%Y-%m-%d %H:%M:%S"),
                user["id"],
            ),
        )

        return {
            "telegram_id": telegram_id,
            "is_premium": True,
            "premium_until": _format_dt(new_premium_until),
        }


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "NutriSnap API",
        "database": "postgresql" if USE_POSTGRES else "sqlite",
    }