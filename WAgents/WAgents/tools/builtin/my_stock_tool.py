# my_stock_tool.py
"""
股票数据查询工具 — 基于 StockDataTool 的 WAgents 工具封装

用法（给 Agent 看的格式）：
    realtime <股票代码>                  — 实时行情
    kline <代码> <开始日期> <结束日期> [周期]  — K线数据
    minute <代码> <日期> [周期]             — 分钟K线
    today <代码>                          — 今日数据摘要
    auction <代码> [日期]                  — 集合竞价数据
    trend <代码> <日期> [周期]             — 日内分时走势

示例：
    realtime 000062
    kline 000062 2025-01-01 2026-06-22 daily
    minute 000062 2026-06-22 5
    today 000062
    auction 000062
    auction 000062 2026-06-23
    trend 000062
    trend 000062 2026-06-23 5
"""

import json
import re
from datetime import datetime, timedelta

from ..registry import ToolRegistry


def _resolve_date(date_str: str) -> str:
    """将日期字符串解析为 YYYY-MM-DD 格式，支持 today/now 等别名"""
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")
    alias = date_str.strip().lower()
    if alias in ("today", "now", "今天", "今日"):
        return datetime.now().strftime("%Y-%m-%d")
    # 已经是 YYYY-MM-DD 或 YYYYMMDD 格式，原样返回
    return date_str


def _parse_command(text: str) -> dict:
    """解析输入文本，返回 {cmd, symbol, args}"""
    text = text.strip()
    if not text:
        return {"cmd": None, "symbol": None, "args": []}

    # 识别命令前缀
    parts = text.split()
    cmd = parts[0].lower()
    known_commands = {"realtime", "kline", "minute", "today", "auction", "trend", "help"}

    if cmd in known_commands:
        return {"cmd": cmd, "symbol": parts[1] if len(parts) > 1 else None, "args": parts[2:]}
    else:
        # 没有命令前缀时，尝试智能推断
        return {"cmd": "auto", "symbol": parts[0], "args": parts[1:]}


def _format_quote(quote: dict) -> str:
    """格式化实时行情为可读文本"""
    lines = [
        f"📊 {quote.get('name', 'N/A')} ({quote.get('code', 'N/A')})",
        f"   最新价: {quote.get('price', 'N/A')}",
        f"   涨跌幅: {quote.get('pct_change', 'N/A')}%",
        f"   涨跌额: {quote.get('change', 'N/A')}",
        f"   今开: {quote.get('open', 'N/A')}",
        f"   最高: {quote.get('high', 'N/A')}",
        f"   最低: {quote.get('low', 'N/A')}",
        f"   昨收: {quote.get('yesterday_close', 'N/A')}",
        f"   成交量: {quote.get('volume', 'N/A')}",
        f"   成交额: {quote.get('amount', 'N/A')}",
    ]
    if quote.get("time"):
        lines.append(f"   时间: {quote.get('date', '')} {quote['time']}")
    return "\n".join(lines)


def _format_kline(df) -> str:
    """格式化K线数据前5条 + 后3条"""
    if df is None or (hasattr(df, 'empty') and df.empty):
        return "未获取到K线数据"

    lines = [f"📈 K线数据共 {len(df)} 条："]

    # 表头
    cols = ["open", "close", "high", "low", "volume"]
    present_cols = [c for c in cols if c in df.columns]
    header = "日期  " + "  ".join(f"{c:>8}" for c in present_cols)
    lines.append(header)

    # 前5条
    count = 0
    for idx, row in df.head(5).iterrows():
        date_str = idx.strftime("%m-%d") if hasattr(idx, "strftime") else str(idx)
        vals = "  ".join(f"{_safe_float(row.get(c, 0)):>8.2f}" if c != "volume" else f"{_safe_float(row.get(c, 0)):>8.0f}" for c in present_cols)
        lines.append(f"{date_str}  {vals}")
        count += 1

    if len(df) > 8:
        lines.append("  ... ...")

    # 后3条
    for idx, row in df.tail(3).iterrows():
        date_str = idx.strftime("%m-%d") if hasattr(idx, "strftime") else str(idx)
        vals = "  ".join(f"{_safe_float(row.get(c, 0)):>8.2f}" if c != "volume" else f"{_safe_float(row.get(c, 0)):>8.0f}" for c in present_cols)
        lines.append(f"{date_str}  {vals}")
        count += 1

    # 均线
    closes_val = [_safe_float(row.get("close")) for _, row in df.iterrows()]
    valid_closes = [c for c in closes_val if not (isinstance(c, float) and c != c)]
    if len(valid_closes) >= 5:
        mas = _calc_mas(valid_closes)
        parts = []
        for n in [5, 10, 20, 60]:
            if n in mas:
                parts.append(f"MA{n}={mas[n]:.2f}")
        # 插入均线到表格最前面（第二条，紧跟总条数之后）
        # 把均线放在最开头，最新价也一并标出
        latest_close = valid_closes[-1] if valid_closes else 0
        ma_parts = [f"收盘价={latest_close:.2f}"]
        for n in [5, 10, 20, 60]:
            if n in mas:
                ma_parts.append(f"MA{n}={mas[n]:.2f}")
        import json as _json
        ma_dict = {}
        for part in ma_parts:
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    ma_dict[k] = float(v)
                except ValueError:
                    ma_dict[k] = v
        lines.insert(0, _json.dumps(ma_dict, ensure_ascii=False))
        del lines[1]  # 移除原来的总条数行
        cur = valid_closes[-1]
        for n in [5, 10, 20]:
            if n in mas:
                pct = (cur - mas[n]) / mas[n] * 100
                flag = "↑" if pct >= 0 else "↓"
                lines.append(f"   价格相对MA{n}: {flag}{abs(pct):.2f}%")
    return "\n".join(lines)


def _format_auction(data: dict) -> str:
    """格式化集合竞价数据为可读文本"""
    lines = [
        f"🔔 集合竞价 — {data.get('name', 'N/A')} ({data.get('symbol', 'N/A')})",
        f"   日期: {data.get('date', 'N/A')}",
        f"   昨收: {data.get('yesterday_close', 'N/A')}",
    ]

    # 撮合结果
    result = data.get("auction_result", {})
    if result:
        lines.extend([
            f"\n🎯 9:25 撮合结果:",
            f"   撮合价: {result.get('price', 'N/A')}",
            f"   涨幅: {result.get('pct_change', 'N/A')}%",
            f"   最高: {result.get('high', 'N/A')}  最低: {result.get('low', 'N/A')}",
        ])
        if result.get("trend"):
            lines.append(f"   走势: {result['trend']}")

    # 竞价过程 (9:15-9:24)
    process = data.get("auction_process", [])
    if process:
        lines.append(f"\n⏳ 竞价过程 ({len(process)} 条):")
        lines.append("  时间      价格    涨幅")
        for p in process:
            lines.append(f"  {p['time']}  {p['price']:>8.2f}  {p.get('pct_change', 0):>+6.2f}%")

    # 开盘后首笔成交
    open_trades = data.get("open_trades", [])
    if open_trades:
        lines.append(f"\n📋 开盘后首笔成交 ({len(open_trades)} 笔):")
        lines.append("  时间      价格    成交量  成交额")
        for t in open_trades:
            lines.append(f"  {t['time']}  {t['price']:>8.2f}  {t.get('volume', 0):>6}手  {t.get('amount', 0):>10.2f}")

    return "\n".join(lines)


def _format_trend(df) -> str:
    """格式化分钟分时走势数据"""
    if df is None or (hasattr(df, 'empty') and df.empty):
        return "未获取到分时数据"

    lines = [f"📉 分时走势共 {len(df)} 条："]
    lines.append("  时间    价格    涨跌%   成交量")

    first_close = _safe_float(df.iloc[0].get("close", 0))
    ref_price = first_close

    for idx, row in df.iterrows():
        time_str = idx.strftime("%H:%M") if hasattr(idx, "strftime") else str(idx)
        price = _safe_float(row.get("close", 0))
        vol = _safe_float(row.get("volume", 0), 0)
        pct = ((price - ref_price) / ref_price * 100) if ref_price else 0
        arrow = "↑" if pct >= 0 else "↓"
        lines.append(f"  {time_str}  {price:>7.2f}  {arrow}{pct:>+6.2f}%  {vol:>8.0f}")

    prices = [_safe_float(row.get("close", 0)) for _, row in df.iterrows()]
    volumes = [_safe_float(row.get("volume", 0), 0) for _, row in df.iterrows()]
    if prices:
        high_idx = prices.index(max(prices))
        low_idx = prices.index(min(prices))
        high_t = df.iloc[high_idx].name.strftime("%H:%M") if hasattr(df.iloc[high_idx].name, "strftime") else str(df.iloc[high_idx].name)
        low_t = df.iloc[low_idx].name.strftime("%H:%M") if hasattr(df.iloc[low_idx].name, "strftime") else str(df.iloc[low_idx].name)
        lines.append(f"\n📊 盘中最高: {max(prices):.2f}（{high_t}）")
        lines.append(f"📊 盘中最低: {min(prices):.2f}（{low_t}）")
        lines.append(f"📊 总成交量: {sum(volumes):.0f}手")

    return "\n".join(lines)


def _calc_mas(closes: list) -> dict:
    """计算各周期均线值"""
    mas = {}
    for n in [5, 10, 20, 60]:
        if len(closes) >= n:
            mas[n] = sum(closes[-n:]) / n
    return mas


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def my_stock(query: str) -> str:
    """
    股票数据查询函数

    支持的查询格式：
        realtime <股票代码>                — 实时行情
        kline <代码> <开始> <结束> [周期]     — K线数据
        minute <代码> <日期> [周期]          — 分钟K线
        today <代码>                       — 今日数据摘要
        auction <代码> [日期]               — 集合竞价数据
        trend <代码> <日期> [周期]           — 日内分时走势

    股票代码支持：000062、000062.SZ、sh600001 等格式。

    Args:
        query: 查询字符串

    Returns:
        格式化的股票数据文本
    """
    try:
        from .stock_data_tool import stock_data_tool
    except ImportError as e:
        return f"❌ 股票数据模块导入失败：{e}，请安装依赖：pip install akshare pandas numpy"

    parsed = _parse_command(query)
    cmd = parsed["cmd"]
    symbol = parsed["symbol"]
    args = parsed["args"]

    if not symbol and cmd != "help":
        return ("📋 股票数据查询工具\n\n"
                "用法：\n"
                "  realtime <代码>               — 实时行情\n"
                "  kline <代码> <开始> <结束> [周期] — K线数据\n"
                "  minute <代码> <日期> [周期]      — 分钟K线\n"
                "  today <代码>                  — 今日数据摘要\n"
                "  auction <代码> [日期]          — 集合竞价数据\n"
                "  trend <代码> <日期> [周期]       — 分时走势（默认5分钟）\n\n"
                "示例：\n"
                "  realtime 000062\n"
                "  kline 000062 2025-01-01 2026-06-22 daily\n"
                "  minute 000062 2026-06-22 5\n"
                "  today 000062\n"
                "  auction 000062\n"
                "  trend 000062")

    if cmd == "help":
        return ("📋 股票数据查询工具\n\n"
                "命令格式：\n"
                "  realtime <代码>                 — 实时行情\n"
                "  kline <代码> <开始> <结束> [周期]  — K线数据\n"
                "  minute <代码> <日期> [周期]       — 分钟K线\n"
                "  today <代码>                    — 今日数据摘要\n"
                "  auction <代码> [日期]            — 集合竞价数据\n"
                "  trend <代码> <日期> [周期]         — 分时走势\n\n"
                "示例：\n"
                "  realtime 000062\n"
                "  kline 000062 2025-01-01 2026-06-22 daily\n"
                "  minute 000062 2026-06-22 5\n"
                "  today 000062\n"
                "  auction 000062\n"
                "  auction 000062 2026-06-23\n"
                "  trend 000062\n"
                "  trend 000062 2026-06-23 5\n\n"
                "股票代码支持：000062、000062.SZ、sh600001 等格式。")

    if cmd in ("realtime", "auto"):
        try:
            quote = stock_data_tool.get_realtime_quote(symbol)
            return _format_quote(quote)
        except Exception as e:
            if cmd == "realtime":
                return f"❌ 获取实时行情失败：{e}"
            # auto 模式：实时行情失败，继续尝试 K线

    # auto 模式下 K线 作为备用
    if cmd in ("kline", "auto"):
        start = _resolve_date(args[0]) if len(args) > 0 else (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        end = _resolve_date(args[1]) if len(args) > 1 else datetime.now().strftime("%Y-%m-%d")
        period = args[2] if len(args) > 2 else "daily"
        try:
            df = stock_data_tool.get_kline(symbol, start, end, period=period)
            return _format_kline(df)
        except Exception as e:
            if cmd == "kline":
                return f"❌ 获取K线数据失败：{e}"
            # auto 模式：K线失败，继续尝试分钟K线

    if cmd in ("minute", "auto"):
        date = _resolve_date(args[0]) if len(args) > 0 else datetime.now().strftime("%Y-%m-%d")
        period = args[1] if len(args) > 1 else "5"
        try:
            df = stock_data_tool.get_minute_kline(symbol, date, period=period)
            return _format_kline(df)
        except Exception as e:
            if cmd == "minute":
                return f"❌ 获取分钟K线失败：{e}"
            # auto 模式全失败
            return f"❌ 未能获取到「{symbol}」的股票数据（实时行情、K线、分钟K线均失败）"

    if cmd == "trend":
        date = _resolve_date(args[0]) if len(args) > 0 else datetime.now().strftime("%Y-%m-%d")
        period = args[1] if len(args) > 1 else "5"
        try:
            df = stock_data_tool.get_minute_kline(symbol, date, period=period)
            return _format_trend(df)
        except Exception as e:
            return f"❌ 获取分时走势失败：{e}"

    if cmd == "auction":
        date = _resolve_date(args[0]) if len(args) > 0 else datetime.now().strftime("%Y-%m-%d")
        try:
            data = stock_data_tool.get_auction_data(symbol, date)
            return _format_auction(data)
        except Exception as e:
            return f"❌ 获取集合竞价数据失败：{e}"

    if cmd == "today":
        try:
            data = stock_data_tool.get_today_data(symbol)
            parts = []
            if data.get("basic"):
                parts.append(_format_quote(data["basic"]))
            if data.get("today"):
                td = data["today"]
                parts.append(
                    f"\n📅 今日K线：\n"
                    f"   开盘 {td.get('open', 'N/A')}  收盘 {td.get('close', 'N/A')}\n"
                    f"   最高 {td.get('high', 'N/A')}  最低 {td.get('low', 'N/A')}\n"
                    f"   成交量 {td.get('volume', 'N/A')}手  成交额 {td.get('amount', 'N/A')}元"
                )
            return "\n\n".join(parts) if parts else "未获取到今日数据"
        except Exception as e:
            return f"❌ 获取今日数据失败：{e}"

    return f"❌ 未知命令「{cmd}」，支持的命令：realtime / kline / minute / today / auction / trend / help"


def create_stock_registry():
    """创建包含股票数据查询功能的工具注册表"""
    registry = ToolRegistry()

    registry.registry_function(
        name="stock_data",
        description=(
            "A股股票数据查询工具。支持实时行情、K线、分钟K线、集合竞价、分时走势等。"
            "输入：realtime <代码> | kline <代码> <开始> <结束> [周期] | minute <代码> <日期> [周期] | today <代码> | auction <代码> [日期] | trend <代码> <日期> [周期]"
        ),
        func=my_stock
    )

    return registry
