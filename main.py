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

def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.convert("L")  # grayscale
    img = img.point(lambda x: 0 if x < 140 else 255)  # binarize
    img = img.filter(ImageFilter.SHARPEN)
    return img

# ---- Config ----
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
DATABASE_URL = os.environ["DATABASE_URL"]

# OCR is optional but supported when Dockerfile installs tesseract
USE_OCR = True
try:
    if USE_OCR:
        import pytesseract
        from PIL import Image
except Exception:
    USE_OCR = False

# ---- DB helpers ----
def get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.DictCursor)

def init_db():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            tx_time TIMESTAMP WITH TIME ZONE,
            source TEXT,                -- 'text' | 'ocr' | other
            action TEXT NOT NULL,       -- 'Buy' or 'Sell'
            symbol TEXT NOT NULL,
            qty NUMERIC(18,6) NOT NULL,
            price NUMERIC(18,6) NOT NULL,
            fee NUMERIC(18,6) NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, COALESCE(tx_time, created_at));
        """)
        conn.commit()

# ---- Utility ----
def parse_keyvals(parts):
    """
    Parse key=value tokens like fee=0.46 at="2025-09-26 13:18"
    Returns dict with lowercase keys.
    """
    out = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out

def parse_time(s: str | None):
    if not s:
        return None
    # support "2025-09-26 13:18", "2025-09-26", "26/09/2025 13:18"
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
    # Make a simple monospace table
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
    """
    trades: list of dict rows, sorted by tx_time/created_at.
    Returns (realized_pnl, open_qty, avg_cost_open)
    Fees: buy fee increases basis; sell fee reduces proceeds.
    """
    buys = deque()  # each item: [qty_remaining, unit_cost (price), fee_per_share]
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

            # If selling more than we own, treat remaining as short opened/closed immediately:
            if remaining > 1e-12:
                # no buys to match -> mark-to-market as if zero basis (conservative); or ignore
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
        "/sum SYMBOL  — list trades + totals\n"
        "/export      — CSV of all trades"
    )

async def add_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /add Buy TSLA 431.89 1 fee=0.46 at=2025-09-26 13:18
    order:   0    1   2     3     4
    """
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
                """INSERT INTO trades (tx_time, source, action, symbol, qty, price, fee)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (tx_time, "text", action, symbol.upper(), float(qty), float(price), fee),
            )
            conn.commit()

        await update.message.reply_text(
            f"✅ Added {action} {qty} {symbol.upper()} @ {price} (fee {fee})"
        )
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {e}")

async def sum_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Usage: /sum SYMBOL")
    symbol = context.args[0].upper()

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT created_at, tx_time, action, symbol, qty, price, fee
               FROM trades
               WHERE symbol=%s
               ORDER BY COALESCE(tx_time, created_at) ASC""",
            (symbol,),
        )
        rows = cur.fetchall()

    if not rows:
        return await update.message.reply_text(f"No trades for {symbol} yet.")

    # Prepare table
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

    # FIFO P&L summary
    realized, open_qty, avg_cost = realized_pnl_fifo(rows)
    table_text = fmt_table(table)
    summary = (
        f"\nRealized P&L: {fmt_money(realized)}\n"
        f"Open Position: {open_qty:.4f} shares @ avg cost ${avg_cost:.2f}"
    )
    
    from telegram.helpers import escape_markdown

    safe_text = escape_markdown(table_text + summary, version=2)
    await update.message.reply_text(safe_text, parse_mode=ParseMode.MARKDOWN_V2)

async def profit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # If symbol provided, compute for that symbol; else for all
    with get_conn() as conn, conn.cursor() as cur:
        if context.args:
            symbol = context.args[0].upper()
            cur.execute(
                """SELECT created_at, tx_time, action, symbol, qty, price, fee
                   FROM trades
                   WHERE symbol=%s
                   ORDER BY COALESCE(tx_time, created_at) ASC""",
                (symbol,),
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
            # All symbols: compute per symbol then total
            cur.execute("""SELECT DISTINCT symbol FROM trades ORDER BY symbol""")
            symbols = [r[0] for r in cur.fetchall()]
            lines = []
            total = 0.0
            for sym in symbols:
                cur.execute(
                    """SELECT created_at, tx_time, action, symbol, qty, price, fee
                       FROM trades WHERE symbol=%s
                       ORDER BY COALESCE(tx_time, created_at) ASC""",
                    (sym,),
                )
                rows = cur.fetchall()
                realized, _, _ = realized_pnl_fifo(rows)
                total += realized
                lines.append(f"{sym}: {fmt_money(realized)}")
            lines.append(f"— — —\nTotal: {fmt_money(total)}")
            await update.message.reply_text("\n".join(lines))
async def profit_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # today = datetime.utcnow().date()  # today in UTC (you can adjust for timezone)
    tz = pytz.timezone("Asia/Bangkok")
    today = datetime.now(tz).date()
    start = datetime.combine(today, datetime.min.time())
    end = datetime.combine(today, datetime.max.time())

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT DISTINCT symbol FROM trades
               WHERE COALESCE(tx_time, created_at) >= %s
                 AND COALESCE(tx_time, created_at) <= %s
               ORDER BY symbol""",
            (start, end),
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
                   WHERE symbol=%s
                     AND COALESCE(tx_time, created_at) >= %s
                     AND COALESCE(tx_time, created_at) <= %s
                   ORDER BY COALESCE(tx_time, created_at) ASC""",
                (sym, start, end),
            )
            rows = cur.fetchall()
            realized, _, _ = realized_pnl_fifo(rows)
            total += realized
            lines.append(f"{sym}: {fmt_money(realized)}")

        lines.append(f"— — —\nToday total: {fmt_money(total)}")
        await update.message.reply_text("\n".join(lines))

async def export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, created_at, tx_time, source, action, symbol, qty, price, fee
               FROM trades ORDER BY COALESCE(tx_time, created_at) ASC"""
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

# ---- OCR from screenshots ----
def ocr_extract(text: str):
    print("📝 OCR RAW:", repr(text))

    action = None
    if re.search(r"\b(Buy|ซื้อ)\b", text, re.I):
        action = "Buy"
    elif re.search(r"\b(Sell|ขาย)\b", text, re.I):
        action = "Sell"

    # Symbol
    m_sym = re.search(r"\b([A-Z]{1,5})\b", text)
    symbol = m_sym.group(1) if m_sym else None

    # Price
    m_price = re.search(r"(?:US\$)?([0-9]+(?:\.[0-9]+)?)", text)
    price = float(m_price.group(1)) if m_price else None

    # Qty
    m_qty = re.search(r"(?:Qty|จำนวน)\s*[:=]?\s*([0-9]+)", text, re.I)
    qty = float(m_qty.group(1)) if m_qty else 1.0

    # Fee (look for "fee 0.46" or "ค่าธรรมเนียม 0.46")
    m_fee = re.search(r"(?:fee|ค่าธรรมเนียม).*?(?:US\$)?([0-9]+\.[0-9]+)", text, re.I)
    fee = float(m_fee.group(1)) if m_fee else 0.0

    # Timestamp (dd/mm/yyyy hh:mm)
    m_time = re.search(r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})", text)
    tx_time = parse_time(m_time.group(1)) if m_time else None

    print(f"✅ OCR PARSED: action={action}, symbol={symbol}, price={price}, qty={qty}, fee={fee}, tx_time={tx_time}")
    return action, symbol, price, qty, fee, tx_time

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not USE_OCR:
        return await update.message.reply_text("OCR not enabled in this build.")
    try:
        photo = update.message.photo[-1]
        file = await photo.get_file()
        byts = await file.download_as_bytearray()

        from PIL import Image
        img = Image.open(io.BytesIO(byts)).convert("RGB")
        # Thai+English if available in your image; Dockerfile installs tesseract + tha+eng
        text = pytesseract.image_to_string(img, lang="eng+tha")

        action, symbol, price, qty, fee, tx_time = ocr_extract(text)

        if not (action and symbol and price):
            return await update.message.reply_text("⚠️ Could not parse screenshot.")

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO trades (tx_time, source, action, symbol, qty, price, fee)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (tx_time, "ocr", action, symbol.upper(), qty, price, fee),
            )
            conn.commit()

        msg = f"📸 OCR logged: {action} {qty:g} {symbol.upper()} @ {price:.2f}"
        if fee: msg += f" (fee {fee:.2f})"
        if tx_time: msg += f" [{tx_time.strftime('%Y-%m-%d %H:%M')}]"
        await update.message.reply_text(msg)

    except Exception as e:
        print("⚠️ OCR exception:", e)
        traceback.print_exc()
        await update.message.reply_text(f"⚠️ OCR error: {e}")
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📖 *Available Commands*\n\n"
        "/start – Welcome message\n"
        "/help – Show this help\n"
        "/add Buy|Sell SYMBOL PRICE QTY [fee=0.00] [at=YYYY-MM-DD HH:MM]\n"
        "/profit [SYMBOL] – Show realized/open P&L\n"
        "/sum SYMBOL – List trades and totals\n"
        "/export – Export all trades to CSV\n\n"
        "You can also send a broker *screenshot*, I'll OCR it 📸"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN_V2)

# ---- App bootstrap ----
def main():
    init_db()
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("add", add_trade))
    app.add_handler(CommandHandler("sum", sum_symbol))
    app.add_handler(CommandHandler("profit", profit))
    app.add_handler(CommandHandler("export", export_csv))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("profit_today", profit_today))

    print("Bot is running...")
    app.run_polling()   # <-- no await

if __name__ == "__main__":
    # start dummy HTTP server in background thread
    threading.Thread(target=run_server, daemon=True).start()

    # run Telegram bot
    main()
