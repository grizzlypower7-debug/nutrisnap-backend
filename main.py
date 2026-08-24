"""
NutriSnap backend — MVP сервер на FastAPI + SQLite.

Функционал:
- Хранит пользователей и их сканы в SQLite.
- POST /scan-meal — принимает фото (файл) или текст, возвращает мок-анализ КБЖУ.
- Лимит: ровно 2 бесплатных скана в сутки на пользователя.
  На 3-й скан за день — 402 Payment Required с сообщением
  "Limit reached. Upgrade to Premium".
"""

import sqlite3
from contextlib import contextmanager
from datetime import date
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DB_PATH = "nutrisnap.db"
FREE_DAILY_LIMIT = 2  # <-- ровно 2 бесплатных скана в день

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
    Позже здесь будет вызов реальной модели (Vision/ASR + LLM).
    """
    return {
        "meal": "Куриное филе с рисом",
        "calories": 450,
        "protein": 40,
        "fat": 6,
        "carbs": 50,
    }


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

        # --- MOCK: тут в будущем будет реальный ИИ-анализ фото/текста ---
        result = mock_analyze_meal()

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
