import operator
from datetime import datetime
from typing import Annotated, Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from passive_goal_creator import PassiveGoalCreator, Goal # type: ignore
from prompt_optimizer import PromptOptimizer, OptimizedGoal # type: ignore
from response_optimizer import ResponseOptimizer # type: ignore
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# Set up environment variables
loaded = load_dotenv()

# ========================================================================= #
#                           PassiveGoalCreator                              #
# ========================================================================= #
class Goal(BaseModel):
    description: str = Field(..., description="目標の説明")

    @property   # Make `obj.text() -> obj.text` 
    def text(self) -> str:
        return f"{self.description}"

class PassiveGoalCreator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(Goal)
    
    def run(self, query: str) -> Goal:
        prompt = ChatPromptTemplate.from_template(
            "ユーザーの入力を分析し、明確で実行可能な目標を生成してください。\n"
            "要件：\n"
            "1. 目標は具体的かつ明確であり、実行可能なレベルで詳細化されている必要があります。\n"
            "2. あなたが実行可能な行動は以下の行動だけです。\n"
            "    - インターネットを利用して、目標を達成するための調査を行う。\n"
            "    - ユーザーのためのレポートを生成する。\n"
            "3. 決して2.以外の行動を取ってはいけません。\n"
            "ユーザーの入力：{query}"
        )
        chain = prompt | self.llm
        return chain.invoke({"query": query}) # type: ignore



# ========================================================================= #
#                          PromptOptimizer                                  #
# ========================================================================= #
class OptimizedGoal(BaseModel):
    desccription: str = Field(..., description="目標の説明")
    metrics: str = Field(..., description="目標の達成度を測定する方法")

    @property
    def text(self) -> str:
        return f"{self.desccription}(測定基準: {self.metrics})"

class PromptOptimizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def run(self, query: str) -> OptimizedGoal:
        prompt = ChatPromptTemplate.from_template(
            "あなたは目標設定の専門家です。以下の目標をSMART原則 (Specific: 具体的、Measurable: 測定可能、Achievable: 達成可能、Relevant: \
            関連性が高い、Time-bound:期限がある) に基づいて最適化してください。\n\n"
            "元の目標:\n"
            "{query}\n\n"
            "指示:\n"
            "1. 元の目標を分析し、不足している要素や改善点を特定してください。\n"
            "2. あなたが実行可能な行動は以下の行動だけです。\n"
            "    - インターネットを利用して、目標を達成するための調査を行う。\n"
            "    - ユーザーのためのレポートを生成する。\n"
            "3. SMART原則の各要素を考慮しながら、目標を具体的かつ詳細に記載してください。\n"
            "    - 一切抽象的な表現を含んではいけません。\n"
            "    - 必ずすべての単語が実行可能かつ具体的であることを確認してください。\n"
            "4. 目標の達成度を測定する方法を具体的かつ詳細に記載してください。\n"
            "5. 元の目標で期限が指定されていない場合は、期限を考慮する必要はありません。\n"
            "6. REMEMBER: 決して2.以外の行動を取ってはいけません。"
        )
        chain = prompt | self.llm.with_structured_output(OptimizedGoal)
        return chain.invoke({"query": query}) # type: ignore


# ========================================================================= #
#                           ResponseOptimizer                               #
# ========================================================================= #
class ResponseOptimizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def run(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはAIエージェントシステムのレスポンス最適化スペシャリストです。\
                    与えられた目標に対して、エージェントが目標にあったレスポンスを返すためのレスポンス仕様を策定してください。"
                ),
                (
                    "human",
                    "以下の手順に従って、レスポンス最適化プロンプトを作成してください:\n\n"
                    "1. 目標分析:\n"
                    "提示された目標を分析し、主要な要素や意図を特定してください。\n\n"
                    "2. レスポンス仕様の策定:\n"
                    "目標達成のための最適なレスポンス仕様を考案してください。トーン、構造、内容の焦点などを考慮に入れてください。\n\n"
                    "3. 具体的な指示の作成:\n"
                    "事前に収集された情報から、ユーザーの期待に沿ったレスポンスをするために必要な、\
                    AIエージェントに対する明確で実行可能な指示を作成してください。あなたの指示によってAIエージェントが実行可能のは、\
                    既に調査済みの結果をまとめることだけです。インターネットへのアクセスはできません。\n\n"
                    "4. 例の提供:\n"
                    "可能であれば、目標に沿ったレスポンスの例を1つ以上含めてください。\n\n"
                    "5. 評価基準の設定:\n"
                    "レスポンスの効果を測定するための基準を定義してください。\n\n"
                    "以下の構造でレスポンス最適化プロンプトを出力してください:\n\n"
                    "目標分析:\n"
                    "[ここに目標の分析結果を記入]\n\n"
                    "レスポンス仕様:\n"
                    "[ここに策定されたレスポンス仕様を記入]\n\n"
                    "AIエージェントへの指示:\n"
                    "[ここにAIエージェントへの具体的な指示を記入]\n\n"
                    "レスポンス例\n"
                    "[ここにレスポンス例を記入]\n\n"
                    "では、以下の目標に対するレスポンス最適化プロンプトを作成してください:\n"
                    "{query}"
                )
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
    
# ========================================================================= #
#                           MultiPathGeneration                             #
# ========================================================================= #
class TaskOption(BaseModel):
    description: str = Field(default="", description="タスクオプションの説明")


class Task(BaseModel):
    task_name: str = Field(..., description="タスクの名前")
    options: list[TaskOption] = Field(default_factory=list, min_items=2, max_items=3, description="2~3個のタスクオプション") # type: ignore


class DecomposedTasks(BaseModel):
    values: list[Task] = Field(default_factory=list, min_items=3, max_items=5, description="3~5個に分解されたタスク") # type: ignore


class MultiPathPlanGenerationState(BaseModel):
    query: str = Field(..., description="ユーザーが入力したクエリ")
    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(default="", description="最適化されたレスポンス")
    tasks: DecomposedTasks = Field(default_factory=DecomposedTasks, description="複数のオプションを持つタスクのリスト") 
    current_task_index: int = Field(default=0, description="現在のタスクのインデックス")
    chosen_options: Annotated[list[int], operator.add] = Field(default_factory=list, description="各タスクで選択されたオプションのインデックス")
    results: Annotated[list[str], operator.add] = Field(default_factory=list, description="実行されたタスクの結果")
    final_output: str = Field(default="", description="最終出力")


class QueryDecomposer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(self, query: str) -> DecomposedTasks:
        prompt = ChatPromptTemplate.from_template(
            f"CURRENT_DATE: {self.current_date}\n"
            "-----\n"
            "タスク: 与えられた目標を3~5個の高レベルタスクに分解し、各タスクに2~3個の具体的なオプションを提供してください。\n"
            "要件:\n"
            "1. 以下の行動だけで目標を達成すること。決して指定された以外の行動を足らないこと。\n"
            "    - インターネットを利用して、目標を達成するための調査を行う。\n"
            "2. 各高レベルタスクは具体的かつ詳細に記載されており、単独で実行ならびに検証可能な情報を含めること。一切抽象的な表現を含まないこと。\n"
            "3. 各項レベルタスクに2~3個の異なるアプローチまたはオプションを提供すること。\n"
            "4. タスクは実行可能な順序でリスク化すること\n"
            "5. タスクは日本語で出力すること。\n\n"
            "REMEMBER: 実行できないタスク、ならびに選択肢は絶対に作成しないでください。\n\n"
            "目標: {query}"
        )
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        return chain.invoke({"query": query}) # type: ignore


class OptionPresenter:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.configurable_fields(max_tokens=ConfigurableField(id="max_tokens"))

    def run(self, task: Task) -> int:
        task_name = task.task_name
        options = task.options

        print(f"\nタスク: {task_name}")
        for i, option in enumerate(options):
            print(f"{i + 1}. {option.description}")

        choice_prompt = ChatPromptTemplate.from_template(
            "タスク: 与えられたタスクとオプションに基づいて、最適なオプションを選択してください。必ず番号のみで回答してください。\n\n"
            "なお、あなたは次の行動しかできません。\n"
            "- インターネットを利用して、目標を達成するための調査を行う。\n"
            "タスク: {task_name}\n"
            "オプション:\n{options_text}\n"
            "選択 (1-{num_options}): "
        )

        options_text = "\n".join(f"{i+1}. {option.description}" for i, option in enumerate(options))
        chain = (
            choice_prompt
            | self.llm.with_config(configurable=dict(max_tokens=1))
            | StrOutputParser()
        )
        choice_str = chain.invoke(
            {
                "task_name": task_name,
                "options_text": options_text,
                "num_options": len(options)
            }
        )
        print(f"==> エージェントの選択: {choice_str}\n")

        return int(choice_str.strip()) - 1


class TaskExecutor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = [TavilySearchResults(max_results=3)]
    
    def run(self, task: Task, chosen_option: TaskOption) -> str:
        agent = create_react_agent(self.llm, self.tools)
        result = agent.invoke(
            {
                "messages": [
                    (
                        "human",
                        f"以下のタスクを実行し、詳細な回答を提供してください:\n\n"
                        f"タスク: {task.task_name}\n"
                        f"選択されたアプローチ: {chosen_option.description}\n\n"
                        f"要件:\n"
                        f"1. 必要に応じて提供されたツールを使用すること。\n"
                        f"2. 実行において徹底的かつ包括的であること。\n"
                        f"3. 可能な限り具体的な事実やデータを提供すること。\n"
                        f"4. 発見事項を明確にまとめること。\n"
                    )
                ]
            }
        )
        return result["messages"][-1].content
    

class ResultAggregator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, query: str, response_definition: str, tasks: list[Task], chosen_options: list[int], results: list[str]) -> str:
        prompt = ChatPromptTemplate.from_template(
            "与えられた目標:\n{query}\n\n"
            "調査結果:\n{task_results}\n\n"
            "与えられた目標に対し、調査結果を用いて、以下の指示に基づいてレスポンスを生成してください。\n"
            "{response_definition}"
        )
        task_results = self._format_task_results(tasks, chosen_options, results)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "query": query,
                "task_results": task_results,
                "response_definition": response_definition
            }
        )
    
    # No `self` is passed; all data must come through parameters. 
    # `@staticmethod` is suited for methods that don't need to access class attributes. 
    @staticmethod
    def _format_task_results(tasks: list[Task], chosen_options: list[int], results: list[str]) -> str:
        task_results = ""
        for i, (task, chosen_option, result) in enumerate(zip(tasks, chosen_options, results)):
            task_name = task.task_name
            chosen_option_desc = task.options[chosen_option].description
            task_results += f"タスク {i+1}: {task_name}\n"
            task_results += f"選択されたアプローチ: {chosen_option_desc}\n"
            task_results += f"結果: {result}\n\n"
        return task_results
            

class MultiPathGeneration:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.passive_goal_creator = PassiveGoalCreator(llm=self.llm)
        self.prompt_optimizer = PromptOptimizer(llm=self.llm)
        self.response_optimizer = ResponseOptimizer(llm=self.llm)
        self.query_decomposer = QueryDecomposer(llm=self.llm)
        self.option_presenter = OptionPresenter(llm=self.llm)
        self.task_executor = TaskExecutor(llm=self.llm)
        self.result_aggregator = ResultAggregator(llm=self.llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        graph = StateGraph(MultiPathPlanGenerationState)
        graph.add_node("goal_setting", self._goal_setting)
        graph.add_node("decompose_query", self._decompose_query)
        graph.add_node("present_options", self._present_options)
        graph.add_node("execute_task", self._execute_task)
        graph.add_node("aggregate_results", self._aggregate_results)
        
        graph.set_entry_point("goal_setting")
        graph.add_edge("goal_setting", "decompose_query")
        graph.add_edge("decompose_query", "present_options")
        graph.add_edge("present_options", "execute_task")

        graph.add_conditional_edges(
            "execute_task",
            lambda state: state.current_task_index < len(state.tasks.values),
            {True: "present_options", False: "aggregate_results"}
        )
        graph.add_edge("aggregate_results", END)

        return graph.compile() # type: ignore

    def _goal_setting(self, state: MultiPathPlanGenerationState) -> dict[str, Any]:
        goal: Goal = self.passive_goal_creator.run(query=state.query)
        optimized_goal: OptimizedGoal = self.prompt_optimizer.run(query=goal.text)
        optimized_response: str = self.response_optimizer.run(query=optimized_goal) # type: ignore
        return {
            "optimized_goal": optimized_goal.text,
            "optimized_response": optimized_response
        }
    
    def _decompose_query(self, state: MultiPathPlanGenerationState) -> dict[str, Any]:
        tasks = self.query_decomposer.run(query=state.optimized_goal)
        return {"tasks": tasks}
    
    def _present_options(self, state: MultiPathPlanGenerationState) -> dict[str, Any]:
        current_task = state.tasks.values[state.current_task_index]
        chosen_option = self.option_presenter.run(task=current_task)
        return {"chosen_options": [chosen_option]}
    
    def _execute_task(self, state: MultiPathPlanGenerationState) -> dict[str, Any]:
        current_task = state.tasks.values[state.current_task_index]
        chosen_option = current_task.options[state.chosen_options[-1]]
        result = self.task_executor.run(task=current_task, chosen_option=chosen_option)
        return {
            "results": [result],
            "current_task_index": state.current_task_index + 1
        }
    
    def _aggregate_results(self, state: MultiPathPlanGenerationState) -> dict[str, Any]:
        final_output = self.result_aggregator.run(
            query=state.optimized_goal,
            response_definition=state.optimized_response,
            tasks=state.tasks.values,
            chosen_options=state.chosen_options,
            results=state.results
        )
        return {"final_output": final_output}
    
    def run(self, query: str) -> str:
        initial_state = MultiPathPlanGenerationState(query=query)
        final_state = self.graph.invoke(initial_state, {"recursion_limit": 1000}) # type: ignore
        return final_state.get("final_output", "最終的な回答の生成に失敗しました。")
    

def main():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = MultiPathGeneration(llm=llm)

    task = "ホラーゲームの作り方"
    result = agent.run(query=task)

    print("\n=== 最終出力 ===")
    print(result)


if __name__ == "__main__":
    main()
    