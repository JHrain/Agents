# test_my_stock_tool.py
"""
股票数据查询工具测试

注意：工具本身返回的均线值是精确计算的，但 deepseek-v4-flash 模型
可能不信任工具数据而自行编造数值，这是模型行为问题，非工具 bug。
"""

from dotenv import load_dotenv
from WAgents.tools.builtin.my_stock_tool import create_stock_registry

load_dotenv()


def test_stock_tool():
    """测试股票数据查询工具（直接验证，不经过 LLM）"""
    registry = create_stock_registry()

    print("🧪 测试股票数据查询工具\n")

    # 测试K线数据 + 均线
    print("--- K线数据（含均线） ---")
    result = registry.execute_tool("stock_data", "kline 000062 2026-01-01 2026-06-22 daily")
    lines = result.split("\n")
    # 第一行是均线 JSON
    print(f"均线数据: {lines[0]}")
    # 验证均线值存在
    assert "MA5" in lines[0], f"MA5 缺失: {lines[0]}"
    assert "MA10" in lines[0], f"MA10 缺失: {lines[0]}"
    assert "MA20" in lines[0], f"MA20 缺失: {lines[0]}"
    print(f"✅ 均线数据完整\n")

    # 测试实时行情
    print("--- 实时行情 ---")
    result = registry.execute_tool("stock_data", "realtime 000062")
    assert "最新价" in result
    print(f"{result}\n")

    # 测试分时走势
    print("--- 分时走势 ---")
    result = registry.execute_tool("stock_data", "trend 000062")
    assert "分时走势" in result
    print(f"✅ 分时数据获取成功\n")

    # 测试集合竞价
    print("--- 集合竞价 ---")
    result = registry.execute_tool("stock_data", "auction 000062")
    assert "集合竞价" in result
    print(f"✅ 集合竞价数据获取成功\n")


def test_with_simple_agent():
    """手动调工具 + LLM 回答（原始方式）"""
    from WAgents.core.llm import WAgents_LLM

    llm = WAgents_LLM()
    registry = create_stock_registry()

    print("🤖 与SimpleAgent集成测试（手动调工具）:")
    user_question = "请分析一下000062今天集合竞价的数据"
    print(f"用户问题: {user_question}")

    stock_result = registry.execute_tool("stock_data", "auction 000062")
    print(f"集合竞价查询结果:\n{stock_result}\n")

    final_messages = [
        {"role": "user", "content": f"集合竞价数据如下:\n{stock_result}\n\n请基于以上数据，用自然语言回答用户的问题:{user_question}"}
    ]

    print("\n🎯 SimpleAgent的回答:")
    response = llm.think(final_messages)
    for chunk in response:
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    # test_stock_tool()
    test_with_simple_agent()
