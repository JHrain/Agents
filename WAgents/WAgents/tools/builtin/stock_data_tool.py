"""
股票数据获取工具 — 统一封装多数据源的实时与历史数据接口

数据源优先级（自动降级）：
    1. curl 直连东方财富 API — 盘前分时 / 分钟K线 / 日线（绕过 Python SSL 问题）
    2. akshare（东方财富封装）— 日线 / 财务 / 批量行情
    3. 新浪财经 — 实时行情 / 分钟K线备用
    4. 腾讯财经 — 分钟K线备用

使用示例：
    from backend.services.stock_data_tool import stock_data_tool

    # 日线K线
    df = stock_data_tool.get_kline('000001', '2025-01-01', '2026-06-22', period='daily')

    # 实时行情
    quote = stock_data_tool.get_realtime_quote('000062')

    # 分钟K线
    df = stock_data_tool.get_minute_kline('000062', '2026-06-22', period='5')

    # 集合竞价数据
    auc = stock_data_tool.get_auction_data('000062', '2026-06-22')

    # 财务数据
    df = stock_data_tool.get_financial_data('000062', data_type='indicator')
"""

import json
import subprocess
import requests
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict, Literal
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)

# ============================================================
# 常量定义
# ============================================================

# 股票代码规范化映射
_MARKET_PREFIX: Dict[str, str] = {
    "6": "sh",       # 上海主板
    "0": "sz",       # 深圳主板
    "3": "sz",       # 创业板
    "4": "bj",       # 北交所
    "8": "bj",       # 北交所
    "68": "sh",      # 科创板
}

# K线周期映射（akshare 参数 → 东方财富 klt 参数）
_PERIOD_EM_MAP: Dict[str, str] = {
    "daily": "101",
    "weekly": "102",
    "monthly": "103",
}

# 分钟周期到新浪 scale 参数的映射
_MINUTE_SINA_SCALE: Dict[str, str] = {
    "5": "5",
    "15": "15",
    "30": "30",
    "60": "60",
}

# 单次请求最大重试次数
_MAX_RETRIES = 2

# 新浪实时行情字段名（按逗号分隔的顺序）
_SINA_REALTIME_FIELDS = [
    "name", "open", "yesterday_close", "price", "high", "low",
    "bid1", "ask1", "volume", "amount",
    "bid1_vol", "bid1_price", "bid2_vol", "bid2_price", "bid3_vol", "bid3_price",
    "bid4_vol", "bid4_price", "bid5_vol", "bid5_price",
    "ask1_vol", "ask1_price", "ask2_vol", "ask2_price", "ask3_vol", "ask3_price",
    "ask4_vol", "ask4_price", "ask5_vol", "ask5_price",
    "date", "time", "status",
]

# HTTP 请求头
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


# ============================================================
# 工具函数
# ============================================================

def _normalize_symbol(symbol: str, style: str = "em") -> str:
    """
    标准化股票代码为指定数据源格式。

    Args:
        symbol: 输入代码，支持 '000001', '000001.SZ', 'sh600001', 'sz000001'
        style: 输出格式
            - 'em': 东方财富格式，如 '0.000001' (深市) 或 '1.600001' (沪市)
            - 'sina': 新浪格式，如 'sz000001' 或 'sh600001'
            - 'eq': 行情页面前缀，如 'sz000001'

    Returns:
        标准化后的股票代码字符串。
    """
    s = symbol.strip().upper()

    # 去掉已有的 .SZ / .SH 后缀
    if s.endswith(".SH"):
        code = s[:-3]
        market = "sh"
    elif s.endswith(".SZ"):
        code = s[:-3]
        market = "sz"
    elif s.startswith("SH"):
        code = s[2:]
        market = "sh"
    elif s.startswith("SZ"):
        code = s[2:]
        market = "sz"
    elif s.startswith("BJ"):
        code = s[2:]
        market = "bj"
    else:
        code = s
        # 根据首位数字推断市场
        if code.startswith("6"):
            market = "sh"
        elif code.startswith(("0", "3")):
            market = "sz"
        elif code.startswith(("4", "8")):
            market = "bj"
        else:
            market = "sz"  # 默认深圳

    if style == "em":
        market_id = "1" if market == "sh" else "0"
        return f"{market_id}.{code}"
    elif style == "sina":
        return f"{market}{code}"
    elif style == "eq":
        return f"{market}{code}"
    else:
        return f"{code}.{'SH' if market == 'sh' else 'SZ'}"


def _safe_float(value, default=np.nan) -> float:
    """安全转换为浮点数。"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value, default=0) -> int:
    """安全转换为整数。"""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _retry_request(url: str, params: dict = None, max_retries: int = _MAX_RETRIES) -> requests.Response:
    """带重试的 HTTP GET 请求。"""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
    raise last_error


# ============================================================
# 主工具类
# ============================================================

class StockDataTool:
    """
    股票数据获取工具 — 统一封装多数据源的实时与历史数据接口。

    支持的数据类型：
        - 历史K线（日线 / 周线 / 月线）
        - 分钟K线（1m / 5m / 15m / 30m / 60m）
        - 集合竞价数据（9:15-9:25 盘前撮合过程）
        - 实时行情（当日价格、成交量等）
        - 财务数据（主要指标 / 利润表 / 资产负债表）
    """

    # ---- 东方财富 API 直连（curl，解决 Python requests SSL 兼容问题） ----

    @staticmethod
    def _fetch_eastmoney_json(url: str, params: dict = None) -> dict:
        """
        使用 curl 子进程直连东方财富 API 并返回 JSON。

        东方财富 push2 / push2his 系列域名在部分 Python 环境中
        存在 SSL 握手兼容性问题（RemoteDisconnected），改用 curl
        可以绕过该问题。

        Args:
            url:    API 地址
            params: URL 查询参数字典

        Returns:
            dict: 解析后的 JSON 数据。
        """
        if params:
            from urllib.parse import urlencode
            url = f"{url}?{urlencode(params)}"

        cmd = [
            "curl", "-s", "-m", "10",
            "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "-H", "Referer: https://quote.eastmoney.com/",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=12)

        if result.returncode != 0:
            raise RuntimeError(f"curl 请求失败: {result.stderr[:200]}")

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"东方财富返回非 JSON 数据: {result.stdout[:200]}") from e

    # ==================== K线数据 ====================

    def get_kline(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        period: Literal["daily", "weekly", "monthly"] = "daily",
        adjust: Literal["qfq", "hfq", ""] = "qfq",
    ) -> pd.DataFrame:
        """
        获取股票历史K线数据。

        支持日线、周线、月线。数据源为东方财富，通过 akshare 调用。

        Args:
            symbol:      股票代码。支持 '000001', '000001.SZ', 'sh600001' 等格式。
            start_date:  起始日期，格式 'YYYYMMDD' 或 'YYYY-MM-DD'。
            end_date:    结束日期，格式 'YYYYMMDD' 或 'YYYY-MM-DD'。
            period:      K线周期。
                         - 'daily'   日线（默认）
                         - 'weekly'  周线
                         - 'monthly' 月线
            adjust:      复权方式。
                         - 'qfq'  前复权（默认）
                         - 'hfq'  后复权
                         - ''     不复权

        Returns:
            pd.DataFrame，包含以下列：
                - date (index):  日期
                - open:          开盘价
                - close:         收盘价
                - high:          最高价
                - low:           最低价
                - volume:        成交量（手）
                - amount:        成交额（元）
                - amplitude:     振幅（%）
                - pct_change:    涨跌幅（%）
                - change:        涨跌额
                - turnover:      换手率（%）

        Example:
            >>> df = tool.get_kline('000001', '2026-06-01', '2026-06-22', period='daily')
            >>> print(df.tail())
        """
        import akshare as ak

        # 标准化日期格式 → YYYYMMDD
        sd = start_date.replace("-", "")
        ed = end_date.replace("-", "")

        # akshare 接受原始代码
        symbol_clean = symbol.strip().upper()
        # 去掉 .SH / .SZ 后缀（akshare 可接受）
        if symbol_clean.endswith((".SH", ".SZ")):
            symbol_clean = symbol_clean[:-3]

        logger.info(f"获取K线: {symbol_clean}, period={period}, {sd}~{ed}")

        # --- 方式1：curl 直连东方财富日线 API（数据最全） ---
        try:
            df = self._get_kline_em_curl(
                symbol=symbol_clean,
                period=period,
                start_date=sd,
                end_date=ed,
                adjust=adjust,
            )
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"curl东方财富日线失败: {e}")

        # --- 方式2：akshare 东方财富 ---
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol_clean,
                period=period,
                start_date=sd,
                end_date=ed,
                adjust=adjust,
            )
            if df is not None and not df.empty:
                return self._normalize_kline_df(df)
        except Exception as e:
            logger.warning(f"akshare东方财富日线失败: {e}")

        # --- 方式3：腾讯备用源 ---
        try:
            logger.warning("东方财富日线全部失败，尝试腾讯备用源…")
            eq_symbol = _normalize_symbol(symbol, style="sina")
            df = ak.stock_zh_a_hist_tx(
                symbol=eq_symbol,
                start_date=sd,
                end_date=ed,
                adjust=adjust,
            )
            if df is not None and not df.empty:
                return self._normalize_kline_df(df)
        except Exception as e:
            pass

        raise RuntimeError(f"K线数据获取失败（所有数据源），股票: {symbol}, 日期: {start_date}~{end_date}")

    def _get_kline_em_curl(
        self,
        symbol: str,
        period: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> Optional[pd.DataFrame]:
        """
        通过 curl 直连东方财富 kline API 获取日/周/月K线。

        Args:
            symbol:     股票代码（纯数字，如 '000062'）
            period:     'daily' | 'weekly' | 'monthly'
            start_date: 起始日期 'YYYYMMDD'
            end_date:   结束日期 'YYYYMMDD'
            adjust:     复权方式

        Returns:
            pd.DataFrame 或 None
        """
        em_secid = _normalize_symbol(symbol, style="em")
        klt = _PERIOD_EM_MAP.get(period, "101")

        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "secid": em_secid,
            "ut": "fa5fd1943c7b386f172d6893dbbf66b3",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": klt,
            "fqt": "1" if adjust == "qfq" else ("2" if adjust == "hfq" else "0"),
            "beg": start_date,
            "end": end_date,
        }
        data = self._fetch_eastmoney_json(url, params)

        if not data.get("data") or not data["data"].get("klines"):
            return None

        records = []
        for line in data["data"]["klines"]:
            parts = line.split(",")
            if len(parts) < 11:
                continue
            # 格式: 日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
            records.append({
                "date": parts[0],
                "open": _safe_float(parts[1]),
                "close": _safe_float(parts[2]),
                "high": _safe_float(parts[3]),
                "low": _safe_float(parts[4]),
                "volume": _safe_float(parts[5], 0),
                "amount": _safe_float(parts[6], 0),
                "amplitude": _safe_float(parts[7]),
                "pct_change": _safe_float(parts[8]),
                "change": _safe_float(parts[9]),
                "turnover": _safe_float(parts[10]),
            })

        if not records:
            return None

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df

    def _normalize_kline_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化K线 DataFrame 列名。"""
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "振幅": "amplitude", "涨跌幅": "pct_change",
            "涨跌额": "change", "换手率": "turnover",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

        return df

    # ==================== 分钟K线 ====================

    def get_minute_kline(
        self,
        symbol: str,
        date: str,
        period: Literal["1", "5", "15", "30", "60"] = "5",
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """
        获取股票日内分钟K线数据。

        数据源优先级：curl 直连东方财富 → 新浪财经 → 腾讯财经。

        Args:
            symbol:  股票代码。支持 '000001', '000001.SZ', 'sh600001' 等格式。
            date:    交易日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'。
            period:  分钟周期。
                     - '1'   1分钟
                     - '5'   5分钟（默认）
                     - '15'  15分钟
                     - '30'  30分钟
                     - '60'  60分钟
            adjust:  复权方式。默认 'qfq'（前复权），东方财富源有效。

        Returns:
            pd.DataFrame，包含以下列：
                - time (index):  时间
                - open:          开盘价
                - close:         收盘价
                - high:          最高价
                - low:           最低价
                - volume:        成交量（手）
                - amount:        成交额（元）

        Example:
            >>> # 获取上午数据
            >>> df = tool.get_minute_kline('000062', '2026-06-22', period='5')
            >>> morning = df[df.index <= '11:30:00']
            >>> print(f"截至11:30成交量: {morning['volume'].sum():,} 手")
        """
        sd = date.replace("-", "") if len(date) > 8 else date

        # --- 方式1：curl 直连东方财富分钟K线（数据最全） ---
        try:
            df = self._get_minute_kline_em_curl(symbol, sd, period, adjust)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"curl东方财富分钟K线失败: {e}")

        # --- 方式2：新浪财经 ---
        try:
            df = self._get_minute_kline_sina(symbol, date, period)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"新浪分钟K线失败: {e}")

        # --- 方式3：腾讯财经 ---
        try:
            df = self._get_minute_kline_tencent(symbol, date, period)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"腾讯分钟K线失败: {e}")

        raise RuntimeError(f"分钟K线获取失败（所有数据源），股票: {symbol}, 日期: {date}")

    def _get_minute_kline_em_curl(
        self, symbol: str, date: str, period: str, adjust: str
    ) -> Optional[pd.DataFrame]:
        """
        通过 curl 直连东方财富 kline API 获取分钟K线。

        Args:
            symbol:  股票代码
            date:    日期 'YYYYMMDD'
            period:  分钟周期 ('1','5','15','30','60')
            adjust:  复权方式

        Returns:
            pd.DataFrame 或 None
        """
        em_secid = _normalize_symbol(symbol, style="em")

        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "secid": em_secid,
            "ut": "fa5fd1943c7b386f172d6893dbbf66b3",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": period,
            "fqt": "1" if adjust == "qfq" else ("2" if adjust == "hfq" else "0"),
            "beg": date,
            "end": date,
            "lmt": "240",
        }
        data = self._fetch_eastmoney_json(url, params)

        if not data.get("data") or not data["data"].get("klines"):
            logger.debug("curl东方财富分钟K线返回空数据")
            return None

        records = []
        for line in data["data"]["klines"]:
            parts = line.split(",")
            if len(parts) < 7:
                continue
            records.append({
                "time": parts[0],
                "open": _safe_float(parts[1]),
                "close": _safe_float(parts[2]),
                "high": _safe_float(parts[3]),
                "low": _safe_float(parts[4]),
                "volume": _safe_float(parts[5], 0),
                "amount": _safe_float(parts[6], 0),
            })

        if not records:
            return None

        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        return df

    def _get_minute_kline_sina(self, symbol: str, date: str, period: str) -> Optional[pd.DataFrame]:
        """
        从新浪财经获取分钟K线。

        Args:
            symbol: 股票代码
            date:   交易日期 'YYYY-MM-DD'
            period: 分钟周期 '5' | '15' | '30' | '60'

        Returns:
            pd.DataFrame 或 None
        """
        sina_symbol = _normalize_symbol(symbol, style="sina")

        url = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
        params = {
            "symbol": sina_symbol,
            "scale": period,
            "datalen": "240",  # 足够覆盖全天
        }
        resp = _retry_request(url, params=params)
        data = resp.json()

        if not data:
            return None

        records = []
        for item in data:
            if item["day"].startswith(date):
                records.append({
                    "time": item["day"],
                    "open": _safe_float(item.get("open")),
                    "close": _safe_float(item.get("close")),
                    "high": _safe_float(item.get("high")),
                    "low": _safe_float(item.get("low")),
                    "volume": _safe_float(item.get("volume"), 0),
                })

        if not records:
            return None

        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        return df

    def _get_minute_kline_tencent(self, symbol: str, date: str, period: str) -> Optional[pd.DataFrame]:
        """
        从腾讯财经获取分钟K线（5分钟）。

        Args:
            symbol: 股票代码
            date:   交易日期 'YYYY-MM-DD'
            period: 分钟周期（腾讯仅支持5分钟）

        Returns:
            pd.DataFrame 或 None
        """
        if period != "5":
            logger.debug("腾讯源仅支持5分钟K线")
            return None

        eq_symbol = _normalize_symbol(symbol, style="eq")
        url = "https://web.ifzq.gtimg.cn/appstock/app/minute/query"
        params = {"_var": "min_data", "code": eq_symbol}

        resp = _retry_request(url, params=params)
        # 腾讯返回格式: min_data={"code":0,"data":{"sz000062":{"data":{"data":["0930 32.30 ...", ...]}}}}
        text = resp.text
        if text.startswith("min_data="):
            text = text[len("min_data="):]

        import json
        data = json.loads(text)
        if data.get("code") != 0:
            return None

        stock_key = list(data.get("data", {}).keys())[0]
        minutes = data["data"][stock_key]["data"]["data"]

        if not minutes:
            return None

        records = []
        for item in minutes:
            parts = item.split()
            if len(parts) >= 2:
                time_str = parts[0]  # "0930"
                price = _safe_float(parts[1])
                vol = _safe_float(parts[2]) if len(parts) > 2 else 0
                fmt_time = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) == 8 else date
                dt_str = f"{fmt_time} {time_str[:2]}:{time_str[2:]}:00"
                records.append({"time": dt_str, "close": price, "volume": vol})

        if not records:
            return None

        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        return df

    def _normalize_minute_df(self, df: pd.DataFrame, source: str = "em") -> pd.DataFrame:
        """标准化分钟K线 DataFrame 列名。"""
        if source == "em":
            col_map = {
                "时间": "time", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time").sort_index()

        return df

    # ==================== 集合竞价数据 ====================

    def get_auction_data(self, symbol: str, date: str) -> Dict:
        """
        获取股票集合竞价盘前分时数据（9:15-9:25）。

        通过 curl 直连东方财富 trends2 接口，可获取从 9:15 开始的
        虚拟撮合过程、9:25 最终撮合结果及开盘后首笔成交数据。

        Args:
            symbol: 股票代码。支持 '000001', '000001.SZ', 'sh600001' 等格式。
            date:   交易日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'。

        Returns:
            Dict，包含以下字段：

            - yesterday_close:  float — 昨日收盘价
            - auction_process:  list[dict] — 9:15-9:24 每分钟虚拟撮合记录，
                               每条含 time/price/high/low/pct_change
            - auction_result:   dict — 9:25 最终撮合结果：
                                · price:       float  撮合价（开盘价）
                                · high:        float  竞价期间最高价
                                · low:         float  竞价期间最低价
                                · pct_change:  float  较昨收涨跌幅(%)
                                · trend:       str    竞价走势描述
            - open_trades:      list[dict] — 开盘后首笔成交(9:26-9:30)，
                               每条含 time/price/volume/amount
            - full_trends:      pd.DataFrame — 全日分时原始数据（9:15-15:00）

        Example:
            >>> auc = tool.get_auction_data('000062', '2026-06-22')
            >>> print(f"集合竞价撮合价: {auc['auction_result']['price']}")
            >>> print(f"竞价过程: {len(auc['auction_process'])} 条记录")
        """
        em_secid = _normalize_symbol(symbol, style="em")
        date_clean = date.replace("-", "") if len(date) > 8 else date

        url = "https://push2his.eastmoney.com/api/qt/stock/trends2/get"
        params = {
            "secid": em_secid,
            "ut": "fa5fd1943c7b386f172d6893dbbf66b3",
            "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
            "ndays": "1",
            "iscr": "1",
            "iscca": "0",
        }

        data = self._fetch_eastmoney_json(url, params)

        if not data.get("data") or not data["data"].get("trends"):
            raise ValueError(f"未获取到 {symbol} {date} 的分时数据")

        trends = data["data"]["trends"]
        name = data["data"].get("name", "")
        yesterday_close = _safe_float(data["data"].get("preClose", 0))

        # --- 解析全日分时 ---
        all_records = []
        for t in trends:
            parts = t.split(",")
            if len(parts) < 8:
                continue
            all_records.append({
                "time": parts[0],
                "open": _safe_float(parts[1]),
                "close": _safe_float(parts[2]),
                "high": _safe_float(parts[3]),
                "low": _safe_float(parts[4]),
                "volume": _safe_float(parts[5], 0),
                "amount": _safe_float(parts[6], 0),
                "latest": _safe_float(parts[7]),
            })

        df_trends = pd.DataFrame(all_records)
        if not df_trends.empty:
            df_trends["time"] = pd.to_datetime(df_trends["time"])
            df_trends = df_trends.set_index("time").sort_index()

        # --- 提取集合竞价过程 (9:15-9:24) ---
        auction_process = []
        auction_start = f"{date_clean[:4]}-{date_clean[4:6]}-{date_clean[6:]} 09:15"
        auction_0924 = f"{date_clean[:4]}-{date_clean[4:6]}-{date_clean[6:]} 09:24"

        for t in trends:
            parts = t.split(",")
            if len(parts) < 8:
                continue
            time_str = parts[0]
            if auction_start <= time_str <= auction_0924:
                price = _safe_float(parts[2])
                pct = ((price - yesterday_close) / yesterday_close * 100) if yesterday_close else 0.0
                auction_process.append({
                    "time": time_str[-5:],
                    "price": price,
                    "high": _safe_float(parts[3]),
                    "low": _safe_float(parts[4]),
                    "pct_change": round(pct, 2),
                })

        # --- 提取 9:25 撮合结果 ---
        auction_result = {}
        auction_0925_prefix = f"{date_clean[:4]}-{date_clean[4:6]}-{date_clean[6:]} 09:25"

        for t in trends:
            parts = t.split(",")
            if parts[0] == auction_0925_prefix:
                price = _safe_float(parts[2])
                high = _safe_float(parts[3])
                low = _safe_float(parts[4])
                pct = ((price - yesterday_close) / yesterday_close * 100) if yesterday_close else 0.0

                # 判断竞价走势
                if auction_process:
                    first_price = auction_process[0]["price"]
                    last_indicated = auction_process[-1]["price"]
                    high_auc = max(r["price"] for r in auction_process)
                    if price < last_indicated:
                        trend_desc = f"尾盘压价（指示价{last_indicated:.2f}→撮合{price:.2f}）"
                    elif price > first_price:
                        trend_desc = f"持续走高（起始{first_price:.2f}→撮合{price:.2f}）"
                    elif high_auc > price:
                        trend_desc = f"冲高回落（最高{high_auc:.2f}→撮合{price:.2f}）"
                    else:
                        trend_desc = "平稳撮合"
                else:
                    trend_desc = ""

                auction_result = {
                    "price": round(price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "pct_change": round(pct, 2),
                    "trend": trend_desc,
                }
                break

        # --- 提取开盘后首笔成交 (9:26-9:30) ---
        open_trades = []
        open_0926 = f"{date_clean[:4]}-{date_clean[4:6]}-{date_clean[6:]} 09:26"
        open_0930 = f"{date_clean[:4]}-{date_clean[4:6]}-{date_clean[6:]} 09:30"

        for t in trends:
            parts = t.split(",")
            if len(parts) < 8:
                continue
            if open_0926 <= parts[0] <= open_0930:
                vol = _safe_float(parts[5], 0)
                amt = _safe_float(parts[6], 0)
                if vol > 0:
                    open_trades.append({
                        "time": parts[0][-5:],
                        "price": _safe_float(parts[2]),
                        "volume": int(vol),
                        "amount": round(amt, 2),
                    })

        return {
            "symbol": symbol.strip().upper().replace(".SH", "").replace(".SZ", ""),
            "name": name,
            "date": date_clean,
            "yesterday_close": round(yesterday_close, 2),
            "auction_process": auction_process,
            "auction_result": auction_result,
            "open_trades": open_trades,
            "full_trends": df_trends,
        }

    # ==================== 实时行情 ====================

    def get_realtime_quote(
        self,
        symbol: Union[str, List[str]],
        source: Literal["em", "sina"] = "em",
    ) -> Union[Dict, pd.DataFrame]:
        """
        获取股票实时行情数据。

        数据源优先级：akshare 东方财富（全市场批量）→ 新浪财经（单只精确）。

        Args:
            symbol: 股票代码或代码列表。
                    单只: '000062' 或 '000062.SZ'
                    多只: ['000001', '000002', '600001']
            source: 数据源选择。
                    - 'em'   东方财富（默认，支持全市场批量，a股全量约5000+）
                    - 'sina' 新浪财经（单只请求，作为备用）

        Returns:
            单只股票时返回 Dict，包含：
                - code:         股票代码
                - name:         股票名称
                - price:        最新价
                - open:         今开盘
                - high:         最高价
                - low:          最低价
                - yesterday_close: 昨收价
                - volume:       成交量（手）
                - amount:       成交额（元）
                - pct_change:   涨跌幅（%）
                - change:       涨跌额
                - amplitude:    振幅（%）
                - turnover:     换手率（%）
                - high_low:     最高/最低
                - time:         数据时间

            多只股票时返回 pd.DataFrame，每行为一只股票，含上述列。

        Example:
            >>> quote = tool.get_realtime_quote('000062')
            >>> print(f"深圳华强 最新价: {quote['price']}, 涨幅: {quote['pct_change']}%")
        """
        if isinstance(symbol, list):
            return self._get_realtime_quote_batch_em(symbol)
        else:
            return self._get_realtime_quote_single(symbol, source=source)

    def _get_realtime_quote_single(self, symbol: str, source: str = "em") -> Dict:
        """获取单只股票实时行情。"""
        if source == "em":
            return self._get_realtime_quote_em_single(symbol)
        else:
            return self._get_realtime_quote_sina(symbol)

    def _get_realtime_quote_em_single(self, symbol: str) -> Dict:
        """从东方财富获取单只股票实时行情。"""
        try:
            import akshare as ak
            df_all = ak.stock_zh_a_spot_em()

            symbol_clean = _normalize_symbol(symbol, style="em").split(".")[-1]
            row = df_all[df_all["代码"] == symbol_clean]

            if row.empty:
                return self._get_realtime_quote_sina(symbol)

            row = row.iloc[0]
            return self._parse_em_spot_row(row)

        except Exception as e:
            logger.warning(f"东方财富实时行情失败: {e}")
            return self._get_realtime_quote_sina(symbol)

    def _get_realtime_quote_batch_em(self, symbols: List[str]) -> pd.DataFrame:
        """从东方财富获取多只股票实时行情。"""
        import akshare as ak

        df_all = ak.stock_zh_a_spot_em()
        codes = [_normalize_symbol(s, style="em").split(".")[-1] for s in symbols]
        df_filtered = df_all[df_all["代码"].isin(codes)]

        records = []
        for _, row in df_filtered.iterrows():
            records.append(self._parse_em_spot_row(row))

        return pd.DataFrame(records)

    def _parse_em_spot_row(self, row: pd.Series) -> Dict:
        """解析东方财富实时行情行。"""
        return {
            "code": str(row.get("代码", "")),
            "name": str(row.get("名称", "")),
            "price": _safe_float(row.get("最新价")),
            "open": _safe_float(row.get("今开")),
            "high": _safe_float(row.get("最高")),
            "low": _safe_float(row.get("最低")),
            "yesterday_close": _safe_float(row.get("昨收")),
            "volume": _safe_int(row.get("成交量")),
            "amount": _safe_float(row.get("成交额")),
            "pct_change": _safe_float(row.get("涨跌幅")),
            "change": _safe_float(row.get("涨跌额")),
            "amplitude": _safe_float(row.get("振幅")),
            "turnover": _safe_float(row.get("换手率")),
        }

    def _get_realtime_quote_sina(self, symbol: str) -> Dict:
        """
        从新浪财经获取实时行情（备用源）。

        新浪单次请求返回完整行情字符串，解析为结构化 Dict。

        Args:
            symbol: 股票代码

        Returns:
            Dict，字段同 get_realtime_quote 单只返回格式。
        """
        sina_symbol = _normalize_symbol(symbol, style="sina")
        url = f"https://hq.sinajs.cn/list={sina_symbol}"
        # 新浪需要 Referer
        headers = {**_HEADERS, "Referer": "https://finance.sina.com.cn/"}

        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = "gb2312"
        text = resp.text

        # 格式: var hq_str_sz000062="数据,...";
        if "=" not in text or '"' not in text:
            raise ValueError(f"新浪返回格式异常: {text[:100]}")

        data_str = text.split('"')[1]
        if not data_str:
            raise ValueError("新浪返回空数据")

        parts = data_str.split(",")
        if len(parts) < 32:
            raise ValueError(f"新浪返回字段不足: {len(parts)}")

        return {
            "code": symbol.strip().upper().replace(".SH", "").replace(".SZ", ""),
            "name": parts[0],
            "open": _safe_float(parts[1]),
            "yesterday_close": _safe_float(parts[2]),
            "price": _safe_float(parts[3]),
            "high": _safe_float(parts[4]),
            "low": _safe_float(parts[5]),
            "volume": _safe_int(parts[8]),
            "amount": _safe_float(parts[9]),
            "date": parts[30],
            "time": parts[31],
            # 计算涨跌幅
            "pct_change": round(
                (_safe_float(parts[3]) - _safe_float(parts[2])) / _safe_float(parts[2]) * 100, 2
            ) if _safe_float(parts[2]) != 0 else 0.0,
            "change": round(_safe_float(parts[3]) - _safe_float(parts[2]), 2),
        }

    # ==================== 财务数据 ====================

    def get_financial_data(
        self,
        symbol: str,
        data_type: Literal["indicator", "profit", "balance"] = "indicator",
    ) -> pd.DataFrame:
        """
        获取股票财务数据。

        数据源为东方财富，通过 akshare 调用。

        Args:
            symbol:    股票代码。支持 '000001', '000001.SZ', 'sh600001' 等格式。
            data_type: 财务数据类型。
                       - 'indicator'  主要财务指标（默认），包含 ROE、ROA、毛利率、
                                      净利率、资产负债率、每股收益、营收增长率等
                       - 'profit'     利润表（年度），包含营业收入、营业利润、净利润、
                                      扣非净利润、每股收益等
                       - 'balance'    资产负债表（年度），包含总资产、总负债、净资产、
                                      流动资产、流动负债等

        Returns:
            pd.DataFrame，具体列名因 data_type 而异（以下列出常用的英文财务字段）。

            indicator（主要财务指标，来自新浪，约140列）常用列：
                - REPORT_DATE:           报告期
                - SECURITY_NAME_ABBR:    股票简称
                - ROEJQ:                 净资产收益率(加权)
                - ROEKCJQ:               扣非净资产收益率(加权)
                - GSOIR:                 销售毛利率
                - GSGGL:                 销售净利率
                - BASIC_EPS:             基本每股收益
                - TOR_YOY:               营业总收入同比增长率
                - PARENTNETP_YOY:        归属净利润同比增长率
                - DAR:                   资产负债率

            profit（利润表年度，来自东方财富，约200列）常用列：
                - REPORT_DATE:           报告期
                - TOTAL_OPERATE_INCOME:  营业总收入
                - OPERATE_PROFIT:        营业利润
                - TOTAL_PROFIT:          利润总额
                - NET_PROFIT:            净利润
                - PARENT_NETPROFIT:      归属母公司净利润
                - DEDUCT_PARENT_NETPROFIT: 扣非净利润
                - BASIC_EPS:             基本每股收益

            balance（资产负债表年度，来自东方财富，约300列）常用列：
                - REPORT_DATE:           报告期
                - TOTAL_ASSETS:          资产总计
                - TOTAL_LIABILITIES:     负债合计
                - TOTAL_EQUITY:          股东权益合计
                - TOTAL_CURRENT_ASSETS:  流动资产合计
                - TOTAL_CURRENT_LIAB:    流动负债合计
                - MONEY_FUNDS:           货币资金
                - ACCOUNTS_RECE:         应收账款

        Example:
            >>> # 获取财务指标
            >>> df = tool.get_financial_data('000062', data_type='indicator')
            >>> print(df[['净资产收益率', '销售毛利率', '每股收益']].tail())
            >>>
            >>> # 获取利润表
            >>> df = tool.get_financial_data('000062', data_type='profit')
            >>> print(df[['营业总收入', '净利润']].tail())
        """
        import akshare as ak

        # 转换为东方财富格式：SH600001 / SZ000001
        symbol_clean = symbol.strip().upper()
        if symbol_clean.endswith(".SH") or symbol_clean.startswith("sh"):
            code = symbol_clean.replace(".SH", "").replace("sh", "").replace("SH", "")
            em_symbol = f"SH{code}"
        elif symbol_clean.endswith(".SZ") or symbol_clean.startswith("sz"):
            code = symbol_clean.replace(".SZ", "").replace("sz", "").replace("SZ", "")
            em_symbol = f"SZ{code}"
        elif symbol_clean.startswith("6"):
            em_symbol = f"SH{symbol_clean}"
        else:
            em_symbol = f"SZ{symbol_clean}"

        # indicator 接口要求 .SZ / .SH 后缀格式（如 '000062.SZ'）
        if symbol_clean.endswith(".SH") or symbol_clean.startswith("sh"):
            code = symbol_clean.replace(".SH", "").replace("sh", "").replace("SH", "")
            dot_symbol = f"{code}.SH"
        elif symbol_clean.endswith(".SZ") or symbol_clean.startswith("sz"):
            code = symbol_clean.replace(".SZ", "").replace("sz", "").replace("SZ", "")
            dot_symbol = f"{code}.SZ"
        elif symbol_clean.startswith("6"):
            dot_symbol = f"{symbol_clean}.SH"
        else:
            dot_symbol = f"{symbol_clean}.SZ"

        logger.info(f"获取财务数据: em={em_symbol}, dot={dot_symbol}, type={data_type}")

        if data_type == "indicator":
            df = ak.stock_financial_analysis_indicator_em(symbol=dot_symbol)
            # 标准化列名（如果包含大量中文，保持原样即可，akshare返回中文列名）
            if "日期" in df.columns:
                df["日期"] = pd.to_datetime(df["日期"])
                df = df.set_index("日期").sort_index()

        elif data_type == "profit":
            df = ak.stock_profit_sheet_by_yearly_em(symbol=em_symbol)
            if "日期" in df.columns:
                df["日期"] = pd.to_datetime(df["日期"])
                df = df.set_index("日期").sort_index()

        elif data_type == "balance":
            df = ak.stock_balance_sheet_by_yearly_em(symbol=em_symbol)
            if "日期" in df.columns:
                df["日期"] = pd.to_datetime(df["日期"])
                df = df.set_index("日期").sort_index()

        else:
            raise ValueError(f"不支持的财务数据类型: {data_type}")

        if df is None or df.empty:
            raise ValueError(f"股票 {symbol} 无财务数据（{data_type}）")

        return df

    # ==================== 便捷查询方法 ====================

    def get_today_data(self, symbol: str) -> Dict:
        """
        快速获取股票今日数据摘要。

        组合日线K线 + 实时行情，返回今日完整快照。

        Args:
            symbol: 股票代码。

        Returns:
            Dict，包含：
                - basic:     实时行情 Dict
                - kline:     近期日线 pd.DataFrame（最近20日）
                - today:     当日日线数据 Dict（如有）

        Example:
            >>> data = tool.get_today_data('000062')
            >>> print(f"最新价: {data['basic']['price']}, "
            ...       f"今日涨幅: {data['basic']['pct_change']}%")
        """
        # 对齐到最近交易日
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

        result = {"basic": None, "kline": None, "today": None}

        # 实时行情
        try:
            result["basic"] = self.get_realtime_quote(symbol, source="sina")
        except Exception as e:
            logger.warning(f"实时行情获取失败: {e}")

        # 近期K线
        try:
            df = self.get_kline(symbol, start_date, end_date, period="daily")
            result["kline"] = df.tail(20)

            # 最新交易日数据
            if not df.empty:
                latest_row = df.iloc[-1]
                result["today"] = {
                    "date": latest_row.name.strftime("%Y-%m-%d")
                    if hasattr(latest_row.name, "strftime")
                    else str(latest_row.name),
                    "open": _safe_float(latest_row.get("open")),
                    "close": _safe_float(latest_row.get("close")),
                    "high": _safe_float(latest_row.get("high")),
                    "low": _safe_float(latest_row.get("low")),
                    "volume": _safe_int(latest_row.get("volume")),
                    "amount": _safe_float(latest_row.get("amount")),
                }
        except Exception as e:
            logger.warning(f"K线数据获取失败: {e}")

        return result

    def get_intraday_summary(
        self,
        symbol: str,
        date: str,
        period: str = "5",
    ) -> Dict:
        """
        获取日内交易摘要，包含上午/下午分段统计。

        Args:
            symbol: 股票代码。
            date:   交易日期 'YYYY-MM-DD'。
            period: 分钟周期，默认 '5'。

        Returns:
            Dict，包含：
                - full_day:       全天分钟DataFrame
                - morning:        上午分钟DataFrame
                - afternoon:      下午分钟DataFrame
                - morning_vol:    上午累计成交量
                - afternoon_vol:  下午累计成交量
                - total_vol:      全日累计成交量
                - open:           开盘价
                - high:           最高价
                - low:            最低价
                - morning_last:   上午收市价

        Example:
            >>> summary = tool.get_intraday_summary('000062', '2026-06-22')
            >>> print(f"11:30价格: {summary['morning_last']}, "
            ...       f"上午成交量: {summary['morning_vol']:,}手")
        """
        df = self.get_minute_kline(symbol, date, period=period)

        if df is None or df.empty:
            raise ValueError(f"无日内数据: {symbol} {date}")

        # 按时间拆分上/下午
        morning = df[df.index.time <= pd.Timestamp("11:30:00").time()].copy()
        afternoon = df[df.index.time >= pd.Timestamp("13:00:00").time()].copy()

        morning_vol = int(morning["volume"].sum()) if "volume" in morning.columns else 0
        afternoon_vol = int(afternoon["volume"].sum()) if "volume" in afternoon.columns else 0

        return {
            "full_day": df,
            "morning": morning,
            "afternoon": afternoon,
            "morning_vol": morning_vol,
            "afternoon_vol": afternoon_vol,
            "total_vol": morning_vol + afternoon_vol,
            "open": _safe_float(df.iloc[0].get("open", df.iloc[0].get("close"))),
            "high": _safe_float(df["high"].max()) if "high" in df.columns else _safe_float(df["close"].max()),
            "low": _safe_float(df["low"].min()) if "low" in df.columns else _safe_float(df["close"].min()),
            "morning_last": _safe_float(morning.iloc[-1]["close"]) if not morning.empty else None,
        }


# ============================================================
# 全局实例
# ============================================================

stock_data_tool = StockDataTool()
