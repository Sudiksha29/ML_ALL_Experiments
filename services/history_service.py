import sqlite3
import json
import datetime
import config

def init_history_db():
    conn = sqlite3.connect(config.HISTORY_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            exp_id TEXT NOT NULL,
            switcher_id TEXT,
            input_summary TEXT NOT NULL,
            prediction_output TEXT NOT NULL,
            confidence REAL,
            duration_ms REAL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(exp_id, switcher_id, inputs, output, confidence=None, duration_ms=0.0):
    try:
        init_history_db()
        conn = sqlite3.connect(config.HISTORY_DB)
        cursor = conn.cursor()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO prediction_history (timestamp, exp_id, switcher_id, input_summary, prediction_output, confidence, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            now,
            exp_id,
            switcher_id or "",
            json.dumps(inputs),
            str(output),
            confidence,
            duration_ms
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[HistoryService] Error saving history: {e}")

def get_recent_history(limit=50):
    try:
        init_history_db()
        conn = sqlite3.connect(config.HISTORY_DB)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "exp_id": r["exp_id"],
                "switcher_id": r["switcher_id"],
                "inputs": json.loads(r["input_summary"]) if r["input_summary"] else {},
                "output": r["prediction_output"],
                "confidence": r["confidence"],
                "duration_ms": r["duration_ms"]
            })
        conn.close()
        return result
    except Exception as e:
        print(f"[HistoryService] Error getting history: {e}")
        return []

def clear_history():
    try:
        init_history_db()
        conn = sqlite3.connect(config.HISTORY_DB)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_history")
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False
