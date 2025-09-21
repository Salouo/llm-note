import operator
from datetime import datetime
from typing import Annotated, Any

from reflection_manager import Reflection, ReflectionManager, TaskReflector
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from passive_goal_creator import Goal, PassiveGoalCreator
from prompt_optimizer import OptimizedGoal, PromptOptimizer
from response_optimizer import ResponseOptimizer
from pydantic import BaseModel, Field


def format_reflections(reflections: list[Reflection]) -> str:
    return (
        "\n\n".join(
            f"<ref_{i}<task>{r.task}</task><reflection>{r.reflection}</reflection></ref_{i}>"
            for i, r in enumerate(reflections)
        )
        if reflections else "No relevant past reflections."
    )


class DecomposedTasks(BaseModel):
    values: list[str] = Field(default_factory=list, min_items=3, max_items=5, description="3~5個に分解されたタスク")


class ReflectiveAgentState(BaseModel):
    query: str = Field(..., description="ユーザーが最初に入力したクエリ")
    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(default="", description="最適化されたレスポンス定義")
    tasks: list[str] = Field(default_factory=list, description="実行するタスクのリスト")
    current_task_index: int = Field(default=0, description="現在実行中のタスクの番号")
    results: Annotated[list[str], operator.add] = Field(default_factory=list, description="実行済みタスクの結果リスト")
    reflection_ids: Annotated[list[str], operator.add] = Field(default_factory=list, description="リフレクション結果のIDリスト")
    final_output: str = Field(default="", description="最終的な出力結果")
    retry_count: int = Field(default=0, description="タスクの再試行回数")


class ReflectiveGoalCreator:
    def __init__(self, llm: ChatOpenAI, relfection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = relfection_manager
        self.passive_goal_creator = PassiveGoalCreator(llm=self.llm)
        self.prompt_optimizer = PromptOptimizer(llm=self.llm)
    
    def run(self, query: str) -> str:
        relevant_reflections = self.reflection_manager.get_relevant_reflections(query=uery)
        reflection_text = format_reflections(relevant_reflections)

        query = f"{query}\n\n目標設定する際に以下の過去に振り返りを考慮すること:\n{reflection_text}"
        goal: Goal = self.passive_goal_creator.run(query=query)
        optimized_goal: OptimizedGoal = self.prompt_optimizer.run(query=goal.text)
        return optimized_goal.text
    

class ReflectionResponseOptmizer:
    def __init__(self, llm: ChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        self.response_optimizer = ResponseOptimizer(llm=llm)
    
    def run(self, query: str) -> str:
        relevant_reflections = self.reflection_manager.get_relevant_reflections(query=query)
        reflection_text = format_reflections(relevant_reflections)
        query = f"{query}\n\nレスポンス最適化に以下の過去の振り返りを考慮すること:\n{reflection_text}"
        optimized_response: str = self.response_optimizer.run(query=query)
        return optimized_response

