"""简单Agent实现 - 基于OpenAI原生API"""


import re
from typing import Iterator, Optional

from ..core.config import Config
from ..core.llm import WAgents_LLM
from ..core.messages import Message

from ..core.agent import Agent


class SimpleAgent(Agent):
    """简单的对话Agent，支持可选的工具调用"""
    def __init__(
        self,
        name: str,
        llm: WAgents_LLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True
    ):
        """
        初始化SimpleAgent

        Args:
            name: Agent名称
            llm: LLM实例
            system_prompt: 系统提示词
            config: 配置对象
            tool_registry: 工具注册表（可选，如果提供则启用工具调用）
            enable_tool_calling: 是否启用工具调用（只有在提供tool_registry时生效）
        """
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None

    def _get_enhanced_system_prompt(self) -> str:
        """构建增强的系统提示词，包含工具信息"""
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        # 获取工具描述
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题：\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式：\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n\n"

        tools_section += "### 参数格式说明\n"
        tools_section += "1. **多个参数**：使用 `key=value` 格式，用逗号分隔\n"
        tools_section += "   示例：`[TOOL_CALL:calculator_multiply:a=12,b=8]`\n"
        tools_section += "   示例：`[TOOL_CALL:filesystem_read_file:path=README.md]`\n\n"
        tools_section += "2. **单个参数**：直接使用 `key=value`\n"
        tools_section += "   示例：`[TOOL_CALL:search:query=Python编程]`\n\n"
        tools_section += "3. **简单查询**：可以直接传入文本\n"
        tools_section += "   示例：`[TOOL_CALL:search:Python编程]`\n\n"

        tools_section += "### 重要提示\n"
        tools_section += "- 参数名必须与工具定义的参数名完全匹配\n"
        tools_section += "- 数字参数直接写数字，不需要引号：`a=12` 而不是 `a=\"12\"`\n"
        tools_section += "- 文件路径等字符串参数直接写：`path=README.md`\n"
        tools_section += "- 工具调用结果会自动插入到对话中，然后你可以基于结果继续回答\n"

        return base_prompt + tools_section

    def _parse_tool_calls(self, text:str) -> list:
        """解析文本中的工具调用"""
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matchs = re.findall(pattern, text)

        tool_call = []

        for tool_name, parameters in matchs:
            tool_call.append({
                'tool_name': tool_name.strip(),
                'parameters': parameters.strip(),
                'original': f'[TOOL_CALL:{tool_name}:{parameters}]'
            })

        return tool_call

    def _excute_tool_call(self, tool_name:str, parameters:str) -> str:
        """执行工具调用"""
        if not self.tool_registry:
            return f"❌ 错误：未配置工具注册表"

        try:
            # 优先查找 Tool 子类对象
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                param_dict = self._parse_tool_parameters(tool_name, parameters)
                result = tool.run(param_dict)
                return f"🔧 工具 {tool_name} 执行结果：\n{result}"

            # 其次查找 registry_function 注册的函数
            func = self.tool_registry.get_function(tool_name)
            if func:
                result = func(parameters)
                return f"🔧 工具 {tool_name} 执行结果：\n{result}"

            return f"❌ 错误：未找到工具 '{tool_name}'"

        except Exception as e:
            return f"❌ 工具调用失败：{str(e)}"

    def _parse_tool_parameters(self, tool_name:str, parameters:str) -> dict:
        """智能解析工具参数"""
        import json
        param_dict = {}

        # 尝试解析JSON格式
        if parameters.strip().startswith('{'):
            try:
                param_dict = json.load(parameters)
                # JSON解析成功，进行类型转换
                param_dict = self._convert_parameter_types(tool_name, param_dict)
                return param_dict
            except json.JSONDecodeError:
                # JSON解析失败，继续使用其他方式
                pass

        if '=' in parameters:
            # 格式: key=value 或 action=search,query=Python
            if ',' in parameters:
                # 多参数
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        param_dict[key.strip()] = value.strip()
            else:
                # 单参数
                key, value = parameters.split('=', 1)
                param_dict[key.strip()] = value.strip()

            # 类型转换
            param_dict = self._convert_parameter_types(tool_name, param_dict)

            # 智能推断action（如果没有指定）
            if 'action' not in param_dict:
                param_dict = self._infer_action(tool_name, param_dict)
        else:
            # 直接传入参数，根据工具类型智能推断
            param_dict = self._infer_simple_parameters(tool_name, parameters)

        return param_dict


    def _convert_parameter_types(self, tool_name: str, param_dict: dict) -> dict:
        """
        根据工具的参数定义转换参数类型
        Args:
            tool_name: 工具名称
            param_dict: 参数字典

        Returns:
            类型转换后的参数字典
        """
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        # 获取工具的参数定义
        try:
            tool_params = tool.get_parameters()
        except:
            return param_dict

        # 创建参数类型映射,参数名作为key，参数类型作为value
        param_types = {}
        for param in tool_params:
            param_types[param.name] = param.type

        # 转换参数类型
        converted_dict = {}
        for key, value in param_dict.items():
            if key in param_types:
                param_type = param_types[key]
                try:
                    if param_type == 'number' or param_type == 'integer':
                        # 转换为数字
                        if isinstance(value, str):
                            converted_dict[key] = float(value) if param_type == 'number' else int(value)
                        else:
                            converted_dict[key] = value
                    elif param_type == 'boolean':
                        # 转换为布尔值
                        if isinstance(value, str):
                            converted_dict[key] = value.lower() in ('true', '1', 'yes')
                        else:
                            converted_dict[key] = bool(value)
                    else:
                        converted_dict[key] = value
                except (ValueError, TypeError):
                    # 转换失败，保持原值
                    converted_dict[key] = value
            else:
                converted_dict[key] = value

        return converted_dict

    def _infer_action(self, tool_name: str, param_dict: dict) -> dict:
        """根据工具类型和参数推断action"""
        if tool_name == 'memory':
            if 'recall' in param_dict:
                param_dict['action'] = 'search'
                param_dict['query'] = param_dict.pop('recall')
            elif 'store' in param_dict:
                param_dict['action'] = 'add'
                param_dict['content'] = param_dict.pop('store')
            elif 'query' in param_dict:
                param_dict['action'] = 'search'
            elif 'content' in param_dict:
                param_dict['action'] = 'add'
        elif tool_name == 'rag':
            if 'search' in param_dict:
                param_dict['action'] = 'search'
                param_dict['query'] = param_dict.pop('search')
            elif 'query' in param_dict:
                param_dict['action'] = 'search'
            elif 'text' in param_dict:
                param_dict['action'] = 'add_text'

        return param_dict

    def _infer_simple_parameters(self, tool_name: str, parameters: str) -> dict:
        """为简单参数推断完整的参数字典"""
        if tool_name == 'rag':
            return {'action': 'search', 'query': parameters}
        elif tool_name == 'memory':
            return {'action': 'search', 'query': parameters}
        else:
            return {'input': parameters}

    def run(self, input_text:str, max_tool_iteration: int = 3, **kwargs) -> str:
        """
        运行SimpleAgent，支持可选的工具调用

        Args：
            input_text: 用户输入
            max_tool_iterations: 最大工具调用迭代次数（仅在启用工具时有效）
            **kwargs: 其他参数

        Returns:
            Agent响应
        """
        # 构建消息列表
        messages = []

        # 添加消息列表
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        # 添加历史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加当前用户信息
        messages.append({"role": "user", "content": input_text})

        # 如果没有启用工具调用，使用原有逻辑
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            return response

        # 迭代处理， 支持多轮工具调用
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iteration:
            # 调用llm
            response = self.llm.invoke(messages, **kwargs)

            # 检查是否有工具调用
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                # 执行工具调用并收集结果
                tool_results = []
                clean_response = response

                # 构建包含工具结果的消息
                messages.append({"role": "assistant", "content": clean_response})

                for call in tool_calls:
                    result = self._excute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    # 从响应中移除工具调用标记
                    clean_response = clean_response.replace(call['original'], "")

                # 添加工具结果
                tool_results_text = "\n\n".join(tool_results)
                messages.append({"role": "user", "content": f"工具执行结果：\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"})

                current_iteration += 1
                continue

            # 没有工具掉用，这是最终回答
            final_response = response
            break

        # 如果超过最大迭代次数，获取最后一次回答
        if current_iteration >= max_tool_iteration and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        # 保存结果到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))

        return final_response

    def stream_run(self, input_text: str, max_tool_iteration: int = 3, **kwargs) -> Iterator[str]:
        """
        流式运行Agent，支持工具调用。最终回答以流式逐 chunk 输出。

        Args:
            input_text: 用户输入
            max_tool_iteration: 最大工具调用迭代次数
            **kwargs: 其他参数

        Yields:
            Agent响应文本片段
        """
        # 构建消息列表
        enhanced_messages = [{"role": "system", "content": self._get_enhanced_system_prompt()}]

        for msg in self._history:
            enhanced_messages.append({"role": msg.role, "content": msg.content})

        enhanced_messages.append({"role": "user", "content": input_text})

        # 如果没有启用工具调用，直接流式输出
        if not self.enable_tool_calling:
            full_response = ""
            for chunk in self.llm.stream_invoke(enhanced_messages, **kwargs):
                full_response += chunk
                yield chunk
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(full_response, "assistant"))
            return

        # 工具调用循环（使用非流式 invoke 来检测 [TOOL_CALL:]）
        current_iteration = 0
        all_tool_results = []

        while current_iteration < max_tool_iteration:
            response = self.llm.invoke(enhanced_messages, **kwargs)
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                break

            # 执行工具调用（不把含 [TOOL_CALL:] 的原始响应加入对话）
            current_results = []
            for call in tool_calls:
                current_results.append(
                    self._excute_tool_call(call["tool_name"], call["parameters"])
                )
            all_tool_results = current_results  # 只保留最近一次工具调用结果
            current_iteration += 1

        # 构建纯净流式上下文（不含工具描述，防止 LLM 重复生成 [TOOL_CALL:]）
        base_prompt = (self.system_prompt or "你是一个有用的AI助手。") + "\n\n重要规则：工具已经返回了精确计算的均线值（如MA5=xxx等），你分析时必须直接使用这些标注值，不得自行重新计算均线。"
        stream_messages = [{"role": "system", "content": base_prompt}]
        stream_messages.append({"role": "user", "content": input_text})

        if all_tool_results:
            results_summary = "\n\n".join(all_tool_results)
            stream_messages.append({"role": "assistant", "content": "好的，我来查询这些数据。"})
            stream_messages.append({
                "role": "user",
                "content": f"工具执行结果如下：\n{results_summary}\n\n请根据以上数据给出完整的回答。"
            })

        # 流式输出最终回答
        self.add_message(Message(input_text, "user"))
        full_response = ""
        for chunk in self.llm.stream_invoke(stream_messages, **kwargs):
            full_response += chunk
            yield chunk

        self.add_message(Message(full_response, "assistant"))
