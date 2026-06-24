# test_my_date_tool.py

from dotenv import load_dotenv
from WAgents.tools.builtin.my_date_tool import create_date_registry

load_dotenv()


def test_date_tool():
    """测试日期工具"""
    registry = create_date_registry()

    print("🧪 测试日期工具\n")

    result = registry.execute_tool("get_today", "")
    print(f"📅 当前日期信息:\n{result}\n")

    assert "2026" in result, f"年份异常: {result}"
    assert "星期" in result, f"星期信息缺失: {result}"
    assert "北京时间" in result, f"时区信息缺失: {result}"
    print("✅ 日期工具测试通过\n")


def test_auto_tool_calling():
    """注入日期+股票工具到 SimpleAgent，让 LLM 自动识别并调用"""
    from WAgents.core.llm import WAgents_LLM
    from WAgents.agents.simple_agent import SimpleAgent
    from WAgents.tools.builtin.my_stock_tool import create_stock_registry

    llm = WAgents_LLM()
    registry = create_date_registry()
    stock_registry = create_stock_registry()

    # 合并工具：将日期工具注册到股票注册表中（共用同一个）
    for tool in stock_registry.get_all_tools():
        registry.registry_tool(tool)
    for name in stock_registry.list_tools():
        if stock_registry.get_function(name):
            func_info = stock_registry._functions[name]
            registry.registry_function(name, func_info["description"], func_info["func"])

    agent = SimpleAgent(
        name="StockAgent",
        llm=llm,
        system_prompt=(
            "你是一个A股股票分析助手。"
            "当需要查询股票数据时，使用 stock_data 工具。"
            "当需要知道当前日期时，使用 get_today 工具。"
        ),
        tool_registry=registry,
    )

    print("🤖 自动工具调用测试（日期 + 股票）：")
    user_question = "今天是几号，帮我查询000062的实时行情"
    print(f"用户问题: {user_question}")
    print()

    for chunk in agent.stream_run(user_question):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    test_date_tool()
    test_auto_tool_calling()
