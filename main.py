import asyncio
import csv
import io
import os
import re
from collections import deque
from datetime import datetime
from server import run_server
import threading
import traceback
import psycopg2
import psycopg2.extras
from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)
import pytz
from PIL import Image, ImageEnhance, ImageFilter
from telegram.helpers import escape_markdown
import requests

# ---- Config ----
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
DATABASE_URL = os.environ["DATABASE_URL"]

# OCR is optional but supported when Dockerfile installs tesseract
USE_OCR = True
try:
    if USE_OCR:
        import pytesseract
except Exception:
    USE_OCR = False

def fetch_stooq_price(symbol_code: str):
    url = f"https://stooq.com/q/l/?s={symbol_code}&f=sd2t2ohlcv&h&e=csv"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    row = next(reader, None)
    if not row:
        return None
    close = row.get("Close")
    if not close or close in {"N/A", "0"}:
        return None
    return float(close)


def get_latest_price(symbol):
    symbol = symbol.strip().lower()
    candidates = [symbol]
    if "." not in symbol:
        candidates.append(f"{symbol}.us")

    for code in candidates:
        try:
            price = fetch_stooq_price(code)
            if price is not None:
                return price
        except Exception as e:
            print(f"⚠️ Price fetch failed for {code}:", e)

    return None

# ---- DB helpers ----
def get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.DictCursor)

def init_db():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            telegram_user_id BIGINT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            tx_time TIMESTAMP WITH TIME ZONE,
            source TEXT,                -- 'text' | 'ocr' | other
            action TEXT NOT NULL,       -- 'Buy' or 'Sell'
            symbol TEXT NOT NULL,
            qty NUMERIC(18,6) NOT NULL,
            price NUMERIC(18,6) NOT NULL,
            fee NUMERIC(18,6) NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_trades_user_symbol_time
            ON trades(telegram_user_id, symbol, COALESCE(tx_time, created_at));
        """)
        conn.commit()

# ---- Utility ----
def parse_keyvals(parts):
    out = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out

def parse_time(s: str | None):
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%d/%m/%Y %H:%M", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def fmt_money(x):
    sign = "+" if x >= 0 else "-"
    return f"{sign}${abs(x):,.2f}"


def fmt_table(rows):
    widths = [max(len(str(col)) for col in colz) for colz in zip(*rows)]
    lines = []
    for i, r in enumerate(rows):
        line = "  ".join(str(c).ljust(widths[j]) for j, c in enumerate(r))
        lines.append(line)
        if i == 0:
            lines.append("-" * len(line))
    return "```\n" + "\n".join(lines) + "\n```"


# ---- FIFO Realized P&L ----
def realized_pnl_fifo(trades):
    buys = deque()
    realized = 0.0
    total_open_qty = 0.0
    total_open_cost = 0.0

    for t in trades:
        action = t["action"].lower()
        qty = float(t["qty"])
        price = float(t["price"])
        fee = float(t["fee"])

        if action == "buy":
            fee_per_share = fee / qty if qty else 0.0
            buys.append([qty, price, fee_per_share])
            total_open_qty += qty
            total_open_cost += qty * (price + fee_per_share)

        elif action == "sell":
            remaining = qty
            fee_per_share_sell = fee / qty if qty else 0.0
            proceeds_per_share = price - fee_per_share_sell

            while remaining > 1e-12 and buys:
                lot_qty, lot_price, lot_fee_ps = buys[0]
                take = min(remaining, lot_qty)

                cost_ps = lot_price + lot_fee_ps
                realized += (proceeds_per_share - cost_ps) * take

                lot_qty -= take
                remaining -= take
                total_open_qty -= take
                total_open_cost -= cost_ps * take

                if lot_qty <= 1e-12:
                    buys.popleft()
                else:
                    buys[0][0] = lot_qty

            if remaining > 1e-12:
                realized += proceeds_per_share * remaining
                remaining = 0.0

    avg_cost_open = (total_open_cost / total_open_qty) if total_open_qty > 1e-12 else 0.0
    return realized, total_open_qty, avg_cost_open


# ---- Command handlers ----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! Send trades or screenshots.\n\n"
        "Commands:\n"
        "/add Buy TSLA 431.89 1 fee=0.46 at=2025-09-26 13:18\n"
        "/profit [SYMBOL]\n"
        "/profit_today – Show today’s realized P&L\n"
        "/sum SYMBOL  — list trades + totals\n"
        "/export      — CSV of all trades\n"
        "/clear SYMBOL|all — remove trades"
    )

async def add_buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Rewrite message to look like /add Buy ...
    if len(context.args) < 3:
        return await update.message.reply_text("Usage:\n/addb SYMBOL PRICE QTY [fee=0.00] [at=YYYY-MM-DD HH:MM]")
    new_text = f"/add Buy {' '.join(context.args)}"
    update.message.text = new_text
    return await add_trade(update, context)


async def add_sell(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Rewrite message to look like /add Sell ...
    if len(context.args) < 3:
        return await update.message.reply_text("Usage:\n/adds SYMBOL PRICE QTY [fee=0.00] [at=YYYY-MM-DD HH:MM]")
    new_text = f"/add Sell {' '.join(context.args)}"
    update.message.text = new_text
    return await add_trade(update, context)

async def add_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        parts = update.message.text.strip().split()
        if len(parts) < 5:
            return await update.message.reply_text(
                "Usage:\n/add Buy|Sell SYMBOL PRICE QTY [fee=0.00] [at=YYYY-MM-DD HH:MM]"
            )
        _, action, symbol, price, qty, *rest = parts
        action = action.capitalize()
        if action not in ("Buy", "Sell"):
            return await update.message.reply_text("Action must be Buy or Sell.")

        kv = parse_keyvals(rest)
        fee = float(kv.get("fee", "0") or 0)
        tx_time = parse_time(kv.get("at"))

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO trades (telegram_user_id, tx_time, source, action, symbol, qty, price, fee)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (user_id, tx_time, "text", action, symbol.upper(), float(qty), float(price), fee),
            )
            conn.commit()

        await update.message.reply_text(
            f"✅ Added {action} {qty} {symbol.upper()} @ {price} (fee {fee})"
        )
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {e}")


async def sum_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        return await update.message.reply_text("Usage: /sum SYMBOL")
    symbol = context.args[0].upper()

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT created_at, tx_time, action, symbol, qty, price, fee
               FROM trades
               WHERE telegram_user_id=%s AND symbol=%s
               ORDER BY COALESCE(tx_time, created_at) ASC""",
            (user_id, symbol),
        )
        rows = cur.fetchall()

    if not rows:
        return await update.message.reply_text(f"No trades for {symbol} yet.")

    table = [("Date", "Action", "Price", "Qty", "Fee")]
    for r in rows:
        dt = r["tx_time"] or r["created_at"]
        table.append((
            dt.strftime("%Y-%m-%d %H:%M"),
            r["action"],
            f"{float(r['price']):.2f}",
            f"{float(r['qty']):.4f}",
            f"{float(r['fee']):.2f}"
        ))

    realized, open_qty, avg_cost = realized_pnl_fifo(rows)
    table_text = fmt_table(table)
    summary = (
        f"\nRealized P&L: {fmt_money(realized)}\n"
        f"Open Position: {open_qty:.4f} shares @ avg cost ${avg_cost:.2f}"
    )
    latest_price = get_latest_price(symbol)
    if latest_price:
        unrealized = (latest_price - avg_cost) * open_qty
        summary += (
            f"\nLatest Price: ${latest_price:.2f}"
            f"\nUnrealized P&L: {fmt_money(unrealized)}"
            f"\nTotal P&L (Realized+Unrealized): {fmt_money(realized + unrealized)}"
        )
    else:
        summary += "\n⚠️ Could not fetch latest price."

    safe_text = escape_markdown(table_text + summary, version=2)
    await update.message.reply_text(safe_text, parse_mode=ParseMode.MARKDOWN_V2)


async def profit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    with get_conn() as conn, conn.cursor() as cur:
        if context.args:
            symbol = context.args[0].upper()
            cur.execute(
                """SELECT created_at, tx_time, action, symbol, qty, price, fee
                   FROM trades
                   WHERE telegram_user_id=%s AND symbol=%s
                   ORDER BY COALESCE(tx_time, created_at) ASC""",
                (user_id, symbol),
            )
            rows = cur.fetchall()
            if not rows:
                return await update.message.reply_text(f"No trades for {symbol} yet.")
            realized, open_qty, avg_cost = realized_pnl_fifo(rows)
            msg = (
                f"{symbol} Realized: {fmt_money(realized)}\n"
                f"Open: {open_qty:.4f} @ ${avg_cost:.2f}"
            )
            return await update.message.reply_text(msg)
        else:
            cur.execute(
                """SELECT DISTINCT symbol FROM trades
                   WHERE telegram_user_id=%s
                   ORDER BY symbol""",
                (user_id,),
            )
            symbols = [r[0] for r in cur.fetchall()]
            lines = []
            total = 0.0
            for sym in symbols:
                cur.execute(
                    """SELECT created_at, tx_time, action, symbol, qty, price, fee
                       FROM trades
                       WHERE telegram_user_id=%s AND symbol=%s
                       ORDER BY COALESCE(tx_time, created_at) ASC""",
                    (user_id, sym),
                )
                rows = cur.fetchall()
                realized, _, _ = realized_pnl_fifo(rows)
                total += realized
                lines.append(f"{sym}: {fmt_money(realized)}")
            lines.append(f"— — —\nTotal: {fmt_money(total)}")
            await update.message.reply_text("\n".join(lines))


async def profit_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    tz = pytz.timezone("Asia/Bangkok")
    today = datetime.now(tz).date()
    start = datetime.combine(today, datetime.min.time())
    end = datetime.combine(today, datetime.max.time())

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT DISTINCT symbol FROM trades
               WHERE telegram_user_id=%s
                 AND COALESCE(tx_time, created_at) >= %s
                 AND COALESCE(tx_time, created_at) <= %s
               ORDER BY symbol""",
            (user_id, start, end),
        )
        symbols = [r[0] for r in cur.fetchall()]
        if not symbols:
            return await update.message.reply_text("No trades today.")

        lines = []
        total = 0.0
        for sym in symbols:
            cur.execute(
                """SELECT created_at, tx_time, action, symbol, qty, price, fee
                   FROM trades
                   WHERE telegram_user_id=%s AND symbol=%s
                     AND COALESCE(tx_time, created_at) >= %s
                     AND COALESCE(tx_time, created_at) <= %s
                   ORDER BY COALESCE(tx_time, created_at) ASC""",
                (user_id, sym, start, end),
            )
            rows = cur.fetchall()
            realized, _, _ = realized_pnl_fifo(rows)
            total += realized
            lines.append(f"{sym}: {fmt_money(realized)}")

        lines.append(f"— — —\nToday total: {fmt_money(total)}")
        await update.message.reply_text("\n".join(lines))


async def export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, created_at, tx_time, source, action, symbol, qty, price, fee
               FROM trades
               WHERE telegram_user_id=%s
               ORDER BY COALESCE(tx_time, created_at) ASC""",
            (user_id,),
        )
        rows = cur.fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "created_at", "tx_time", "source", "action", "symbol", "qty", "price", "fee"])
    for r in rows:
        writer.writerow([
            r["id"],
            r["created_at"],
            r["tx_time"] or "",
            r["source"] or "",
            r["action"],
            r["symbol"],
            float(r["qty"]),
            float(r["price"]),
            float(r["fee"]),
        ])
    output.seek(0)
    await update.message.reply_document(
        document=InputFile(io.BytesIO(output.getvalue().encode("utf-8")), filename="trades.csv"),
        caption="All trades (CSV)"
    )


async def clear_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        return await update.message.reply_text("Usage: /clear SYMBOL|all")

    target = context.args[0].strip().lower()
    delete_query = """DELETE FROM trades WHERE telegram_user_id=%s"""
    params = [user_id]

    if target != "all":
        delete_query += " AND symbol=%s"
        params.append(target.upper())

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(delete_query, params)
        deleted = cur.rowcount
        conn.commit()

    if deleted == 0:
        if target == "all":
            return await update.message.reply_text("No trades to remove.")
        return await update.message.reply_text(f"No trades found for {target.upper()}.")

    if target == "all":
        msg = f"🧹 Removed all trades ({deleted} rows)."
    else:
        msg = f"🧹 Removed {deleted} trades for {target.upper()}."
    await update.message.reply_text(msg)


# ---- OCR from screenshots ----
def ocr_extract(text: str):
    print("📝 OCR RAW:", repr(text))

    action = None
    if re.search(r"\b(Buy|ซื้อ)\b", text, re.I):
        action = "Buy"
    elif re.search(r"\b(Sell|ขาย)\b", text, re.I):
        action = "Sell"

    m_sym = re.search(r"\b([A-Z]{1,5})\b", text)
    symbol = m_sym.group(1) if m_sym else None

    # m_price = re.search(r"(?:US\$)?([0-9]+(?:\.[0-9]+)?)", text)
    m_price = re.search(r"([0-9]+[.,][0-9]{2})", text.replace(" ", ""))

    price = float(m_price.group(1)) if m_price else None

    m_qty = re.search(r"(?:Qty|จำนวน)\s*[:=]?\s*([0-9]+)", text, re.I)
    qty = float(m_qty.group(1)) if m_qty else 1.0

    m_fee = re.search(r"(?:fee|ค่าธรรมเนียม).*?(?:US\$)?([0-9]+\.[0-9]+)", text, re.I)
    fee = float(m_fee.group(1)) if m_fee else 0.0

    m_time = re.search(r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})", text)
    tx_time = parse_time(m_time.group(1)) if m_time else None

    print(f"✅ OCR PARSED: action={action}, symbol={symbol}, price={price}, qty={qty}, fee={fee}, tx_time={tx_time}")
    return action, symbol, price, qty, fee, tx_time

def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.convert("L")  # grayscale
    img = img.point(lambda x: 0 if x < 140 else 255)  # binarize
    img = img.filter(ImageFilter.SHARPEN)
    return img
def clean_ocr_text(text: str) -> str:
    # remove duplicate spaces
    text = re.sub(r"\s+", " ", text)
    # remove stray symbols like < or FG
    text = re.sub(r"[^A-Za-z0-9ก-๙\.\s:/]", "", text)
    return text.strip()

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not USE_OCR:
        return await update.message.reply_text("OCR not enabled in this build.")
    try:
        user_id = update.effective_user.id
        photo = update.message.photo[-1]
        file = await photo.get_file()
        byts = await file.download_as_bytearray()

        # Apply preprocessing before OCR
        tmp_path = "/tmp/ocr_input.png"
        with open(tmp_path, "wb") as f:
            f.write(byts)

        img = preprocess_image(tmp_path)  # <-- grayscale + binarize + sharpen
        img = img.resize((img.width*2, img.height*2), Image.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        # text = pytesseract.image_to_string(img, lang="eng+tha")
        # text = clean_ocr_text(text)
        data = pytesseract.image_to_data(
        img, output_type=pytesseract.Output.DICT, lang="eng")

        # Collect words with reasonable confidence
        words = []
        for i, word in enumerate(data["text"]):
            if word.strip() and int(data["conf"][i]) > 60:  # filter out junk
                words.append(word)

        text = " ".join(words)
        print("📝 OCR FILTERED:", text)

        # Also reply raw OCR to Telegram for debugging
        await update.message.reply_text(f"📝 OCR words (conf>60):\n{text}")


        # Show raw OCR result to user (escaped for Markdown)
        safe_text = escape_markdown(text.strip() or "(empty)", version=2)
        await update.message.reply_text(f"📝 OCR Text:\n```\n{safe_text}\n```", parse_mode=ParseMode.MARKDOWN_V2)

        action, symbol, price, qty, fee, tx_time = ocr_extract(text)

        if not (action and symbol and price):
            print(f"⚠️ Could not parse screenshot from user {user_id}: {text}")
            return await update.message.reply_text(f"⚠️ Could not parse screenshot from user {user_id}: {text}")

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO trades (telegram_user_id, tx_time, source, action, symbol, qty, price, fee)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (user_id, tx_time, "ocr", action, symbol.upper(), qty, price, fee),
            )
            conn.commit()

        msg = f"📸 OCR logged: {action} {qty:g} {symbol.upper()} @ {price:.2f}"
        if fee:
            msg += f" (fee {fee:.2f})"
        if tx_time:
            msg += f" [{tx_time.strftime('%Y-%m-%d %H:%M')}]"
        await update.message.reply_text(msg)

    except Exception as e:
        print("⚠️ OCR exception:", e)
        traceback.print_exc()
        await update.message.reply_text(f"⚠️ OCR error: {e}")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📖 *Available Commands*\n\n"
        "\/start – Welcome message\n"
        "\/help – Show this help\n"
        "\/add Buy|Sell SYMBOL PRICE QTY [fee=0.00] [at=YYYY-MM-DD HH:MM]\n"
        "\/addb SYMBOL PRICE QTY [fee=...] – Quick *Buy*\n"
        "\/adds SYMBOL PRICE QTY [fee=...] – Quick *Sell*\n"
        "\/profit [SYMBOL] – Show realized\/open P&L\n"
        "\/profit\_today – Show today’s realized P&L\n"
        "\/sum SYMBOL – List trades and totals\n"
        "\/export – Export all trades to CSV\n"
        "\/clear SYMBOL|all – Remove trades\n\n"
        "You can also send a broker *screenshot*, I'll OCR it 📸"
    )
    # await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN_V2)
    await update.message.reply_text(
    msg.replace("*", "<b>").replace("*", "</b>"),
    parse_mode=ParseMode.HTML)


# ---- App bootstrap ----
def main():
    init_db()
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("add", add_trade))
    app.add_handler(CommandHandler("addb", add_buy))
    app.add_handler(CommandHandler("adds", add_sell))
    app.add_handler(CommandHandler("sum", sum_symbol))
    app.add_handler(CommandHandler("profit", profit))
    app.add_handler(CommandHandler("profit_today", profit_today))
    app.add_handler(CommandHandler("export", export_csv))
    app.add_handler(CommandHandler("clear", clear_trades))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"^addb "), add_buy))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"^adds "), add_sell))

    print("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    threading.Thread(target=run_server, daemon=True).start()
    main()
