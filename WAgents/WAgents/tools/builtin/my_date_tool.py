# my_date_tool.py
"""
日期查询工具 — 告诉大模型今天的日期

大模型本身不知道当前日期，此工具让 LLM 能获取准确日期，
便于在查询股票等需要日期的场景中正确理解"今天"。
"""

from datetime import datetime, timezone, timedelta

from ..registry import ToolRegistry

# 北京时间时区
_BJT = timezone(timedelta(hours=8))


def my_today(query: str) -> str:
    """
    获取当前日期和时间信息。

    参数含义：
        仅支持 presence 输入，无实际参数。

    Returns:
        包含当前日期、时间、星期等信息的字符串
    """
    now = datetime.now(_BJT)
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[now.weekday()]

    return (
        f"📅 当前日期：{now.strftime('%Y-%m-%d')}\n"
        f"⏰ 当前时间：{now.strftime('%H:%M:%S')}\n"
        f"📆 星期：{weekday}\n"
        f"🌏 时区：北京时间 (UTC+8)"
    )


def create_date_registry():
    """创建包含日期查询功能的工具注册表"""
    registry = ToolRegistry()

    registry.registry_function(
        name="get_today",
        description="获取今天的当前日期、时间和星期。无需参数，直接输入任何文本即可返回今日日期。",
        func=my_today
    )

    return registry
