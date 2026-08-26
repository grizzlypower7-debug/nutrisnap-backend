"""
NutriSnap backend — MVP сервер на FastAPI + SQLite.

Функционал:
- Хранит пользователей и их сканы в SQLite.
- POST /scan-meal — принимает фото (файл) или текст, анализирует блюдо
  через gpt-4o-mini (ProxyAPI) и возвращает КБЖУ.
- Лимит: ровно 2 бесплатных скана в сутки на пользователя.
  На 3-й скан за день — 402 Payment Required с сообщением
  "Limit reached. Upgrade to Premium".
- Если ИИ недоступен (нет ключа, сбой сети, невалидный ответ) — сервер
  никогда не падает и тихо откатывается на тестовый mock-результат.
"""

import base64
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

DB_PATH = "nutrisnap.db"
FREE_DAILY_LIMIT = 2  # <-- ровно 2 бесплатных скана в день

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
# База данных
# ---------------------------------------------------------------------------

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id TEXT UNIQUE NOT NULL,
                daily_limit INTEGER NOT NULL DEFAULT 2,
                is_premium INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                scan_date TEXT NOT NULL,   -- YYYY-MM-DD, для подсчёта дневного лимита
                input_type TEXT NOT NULL,  -- 'photo' | 'text'
                meal_name TEXT,
                calories INTEGER,
                protein INTEGER,
                fat INTEGER,
                carbs INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )


@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def get_or_create_user(conn: sqlite3.Connection, telegram_id: str) -> sqlite3.Row:
    user = conn.execute(
        "SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)
    ).fetchone()
    if user is None:
        conn.execute(
            "INSERT INTO users (telegram_id, daily_limit) VALUES (?, ?)",
            (telegram_id, FREE_DAILY_LIMIT),
        )
        user = conn.execute(
            "SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)
        ).fetchone()
    return user


def count_scans_today(conn: sqlite3.Connection, user_id: int) -> int:
    today = date.today().isoformat()
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM scans WHERE user_id = ? AND scan_date = ?",
        (user_id, today),
    ).fetchone()
    return row["cnt"]


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
                            "image_url": {"url": data_url},
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

        # Премиум-пользователей лимит не касается
        if not user["is_premium"]:
            used_today = count_scans_today(conn, user["id"])
            if used_today >= FREE_DAILY_LIMIT:
                raise HTTPException(
                    status_code=402,
                    detail="Limit reached. Upgrade to Premium",
                )

        # --- Реальный ИИ-анализ (gpt-4o-mini через ProxyAPI), с fallback на mock ---
        if photo is not None:
            image_bytes = await photo.read()
            result = analyze_meal_photo(image_bytes, photo.content_type)
        else:
            result = analyze_meal_text(text)

        input_type = "photo" if photo is not None else "text"
        today = date.today().isoformat()

        conn.execute(
            """
            INSERT INTO scans
                (user_id, scan_date, input_type, meal_name, calories, protein, fat, carbs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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

        used_after = count_scans_today(conn, user["id"])
        left = max(FREE_DAILY_LIMIT - used_after, 0) if not user["is_premium"] else -1

        return ScanResult(
            **result,
            scans_used_today=used_after,
            scans_left_today=left,
        )


@app.get("/scan-meal/status")
def scan_status(telegram_id: str):
    """Узнать, сколько сканов осталось сегодня, не тратя лимит."""
    with get_db() as conn:
        user = get_or_create_user(conn, telegram_id)
        used = count_scans_today(conn, user["id"])
        left = max(FREE_DAILY_LIMIT - used, 0) if not user["is_premium"] else -1
        return {
            "telegram_id": telegram_id,
            "is_premium": bool(user["is_premium"]),
            "scans_used_today": used,
            "scans_left_today": left,
            "daily_limit": FREE_DAILY_LIMIT,
        }


@app.get("/")
def root():
    return {"status": "ok", "service": "NutriSnap API"}