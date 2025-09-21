import operator
from typing import Annotated, Any, Optional

from dotenv import load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


# Set up environment variables
loaded = load_dotenv()


# `Persona` data model
class Persona(BaseModel):
    name: str = Field(..., description="ペルソナの名前")
    background: str = Field(..., description="ペルソナの持つ背景")


# `Persona` list data model
class Personas(BaseModel):
    personas: list[Persona] = Field(default_factory=list, description="ペルソナのリスト")


# `interview` content data model
class Interview(BaseModel):
    persona: Persona = Field(..., description="インタビュー対象のペルソナ")
    question: str = Field(..., description="インタビューでの質問")
    answer: str = Field(..., description="インタビューでの回答")


# `interview` result data model
class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(default_factory=list, description="インタビュー結果のリスト")


# `evaluation` result data model
class EvaluationResult(BaseModel):
    is_sufficient: bool = Field(..., description="情報が十分かどうか")
    reason: str = Field(..., description="判断の理由")


# state of agent
class InterviewState(BaseModel):
    user_request: str = Field(..., description="ユーザーからのリクエスト")
    personas: Annotated[list[Persona], operator.add] = Field(default_factory=list, description="実施されたインタビューのリスト")
    interviews: Annotated[list[Interview], operator.add] = Field(default_factory=list, description="実施されたインタビューのリスト")
    requirements_doc: str = Field(default="", description="生成された要件定義")
    iteration: int = Field(default=0, description="ペルソナ生成とインタビューの反復回数")
    is_information_sufficient: bool = Field(default=False, description="情報が十分かどうか")


# Class used to generate `persona`
class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 5):
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        # Define prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "あなたはユーザーインタビュー用の多様なペルソナを作成する専門家です。"),
                (
                    "human", 
                    f"以下のユーザーリクエストに関するインタビュー用に、{self.k}人の多様なペルソナを生成してください。\n\n"
                    "ユーザーリクエスト: {user_request}\n\n"
                    "各ペルソナに名前と簡単な背景を含めてください。年齢、性別、職業、技術的専門知識において多様性を確保してください。"
                )
            ]
        )
        
        # Create chain used to generate `persona`
        chain = prompt | self.llm
        # Generate `persona`
        return chain.invoke({"user_request": user_request})
    

# Class for implementing the interview
class InterviewConductor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _generate_questions(self, user_request: str, personas: list[Persona]) -> list[str]:
        # Define prompt for generating questions
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはユーザー要件にも基づいて適切な質問を生成する専門家です。"
                ),
                (
                    "human",
                    "以下のペルソナに関連するユーザーリクエストについて、一つの質問を生成してください。\n\n"
                    "ユーザーリクエスト：{user_request}\n"
                    "ペルソナ：{persona_name} - {persona_background}\n\n"
                    "質問は具体的で、このペルソナの視点から重要な情報を引き出すように設計してください。"
                )
            ]
        )

        # Create chain for genreating quesitons
        question_chain = question_prompt | self.llm | StrOutputParser()
        # Create question list for each persona
        # Each key-value will be embeded into question_prompt
        question_queries = [
            {
                "user_request": user_request,
                "persona_name": persona.name,
                "persona_background": persona.background
            }
            for persona in personas
        ]
        # Generate questions for each persona
        return question_chain.batch(question_queries)
    
    def _generate_answers(self, personas: list[Persona], questions: list[str]) -> list[str]:
        # Define prompt for generating answers corresponding to questions
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは以下のペルソナとして回答しています：{persona_name} - {persona_background}",
                ),
                ("human", "質問：{question}")
            ]
        )

        # Create chain for generating answers
        answer_chain = answer_prompt | self.llm | StrOutputParser()

        # Create list to store answers
        answer_queries = [
            {
                "persona_name": persona.name,
                "persona_background": persona.background,
                "question": question
            }
            for persona, question in zip(personas, questions)
        ]
        # Use `chain.batch` to generate a list of answers
        return answer_chain.batch(answer_queries)
    
    def _create_interviews(self, personas: list[Persona], questions: list[str], answers: list[str]) -> list[Interview]:
        # Generate `Interview` object including persona's questions and answers
        return [
            Interview(persona=persona, question=question, answer=answer)
            for persona, question, answer in zip(personas, questions, answers)
        ]
    
    def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
        # Generate questions
        questions = self._generate_questions(
            user_request=user_request, personas=personas
        )
        # Generate answers
        answers = self._generate_answers(personas=personas, questions=questions)
        # Generate a list containing `Interview` objects
        interviews = self._create_interviews(personas=personas, questions=questions, answers=answers)
        # return results of interviews
        return InterviewResult(interviews=interviews)
    

# Class used for evaluating information's sufficiency
class InformationEvaluator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(EvaluationResult)

    def run(self, user_request: str, interviews: list[Interview]) -> EvaluationResult:
        # Define prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは包括的な要件文書を作成するための情報の十分性を評価する専門家です。"
                ),
                (
                    "human",
                    "以下のユーザーリクエストとインタビュー結果に基づいて、包括的な要件文書を作成するのに十分な情報が集まってたかどうかを判断してください。\n\n"
                    "ユーザーリクエスト：{user_request}\n\n"
                    "インタビュー結果：\n{interview_results}"
                )
            ]
        )

        # Create chain for generating evaluation
        chain = prompt | self.llm
        # Return evaluation result
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"ペルソナ：{interview.persona.name} - {interview.persona.background}\n"
                    f"質問：{interview.question}\n回答：{interview.answer}\n"
                    for interview in interviews
                )
            }
        )


# Class used for generating requirement document
class RequirementDocumentGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        # Define prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは収集した情報に基づいて要件文書を作成する専門家です。"
                ),
                (
                    "human",
                    "以下のユーザーリクエストと複数のペルソナからのインタビュー結果に基づいて、要件文書を作成してください。\n\n"
                    "ユーザーリクエスト：{user_request}\n\n"
                    "インタビュー結果：\n{interview_results}\n"
                    "要件文書には以下のsクションを含めてください：\n"
                    "1. プロジェクト概要\n"
                    "2. 主要機能\n"
                    "3. 非機能要件\n"
                    "4. 制約条件\n"
                    "5. ターゲットユーザー\n"
                    "6. 優先順位\n"
                    "7. リスクと軽減策\n\n"
                    "出力は必ず日本語でお願いします。\n\n要件文書:"
                )
            ]
        )
        # Create chain for generating requirement document
        chain = prompt | self.llm | StrOutputParser()
        # Return requirement document
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"ペルソナ：{interview.persona.name} - {interview.persona.background}\n"
                    f"質問：{interview.question}\n回答：{interview.answer}\n"
                    for interview in interviews
                )
            }
        )


# AI agent class for Requirement document generator
class DocumentationAgent:
    def __init__(self, llm: ChatOpenAI, k: int = 5):
        # Initialize  generators
        self.persona_generator = PersonaGenerator(llm=llm, k=k)
        self.interview_conductor = InterviewConductor(llm=llm)
        self.information_evaluator = InformationEvaluator(llm=llm)
        self.requirements_generator = RequirementDocumentGenerator(llm=llm)
    
        # Create graph
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        # Initialize graph
        workflow = StateGraph(InterviewState)

        # Add nodes
        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("evaluate_information", self._evaluate_information)
        workflow.add_node("generate_requirements", self._generate_requirements)

        # Set entry point
        workflow.set_entry_point("generate_personas")

        # Add edges
        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "evaluate_information")

        # Add conditional edges
        workflow.add_conditional_edges(
            "evaluate_information",
            lambda state: not state.is_information_sufficient and state.iteration < 5,
            {True: "generate_personas", False: "generate_requirements"}
        )
        workflow.add_edge("generate_requirements", END)
        
        # Compile the graph
        return workflow.compile()
    
    def _generate_personas(self, state: InterviewState) -> dict[str, Any]:
        # Generate persona
        new_personas: Personas = self.persona_generator.run(state.user_request)
        return {
            "personas": new_personas.personas,
            "iteration": state.iteration + 1
        }
    
    def _conduct_interviews(self, state: InterviewState) -> dict[str, Any]:
        # Implement interviews
        new_interviews: InterviewResult = self.interview_conductor.run(state.user_request, state.personas[-5:])
        return {"interviews": new_interviews.interviews}
    
    def _evaluate_information(self, state: InterviewState) -> dict[str, Any]:
        # Evaluate information
        evaluation_result: EvaluationResult = self.information_evaluator.run(state.user_request, state.interviews)
        return {
            "is_information_sufficient": evaluation_result.is_sufficient,
            "evaluation_reason": evaluation_result.reason
        }
    
    def _generate_requirements(self, state: InterviewState) -> dict[str, Any]:
        # Generate requirement document
        requiments_doc: str = self.requirements_generator.run(state.user_request, state.interviews)
        return {"requirements_doc": requiments_doc}
    
    def run(self, user_request: str) -> str:
        initial_state = InterviewState(user_request=user_request)
        final_state = self.graph.invoke(initial_state) # type: ignore
        return final_state["requirements_doc"]
    

def main():
    import argparse

    # Initialize parser object
    parser = argparse.ArgumentParser(
        description="ユーザー要求に基づいて要件定義を生成します"
    )

    # Add user_request
    parser.add_argument(
        "--task",
        type=str,
        help="作成したいアプリケーションについて記載してください"
    )

    # Add number of persona
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="生成するペルソナの人数を設定してください (デフォルト:5)"
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Define LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Instantiate documentation agent
    agent = DocumentationAgent(llm=llm, k=args.k)

    # Run the agnet and get the result
    final_output = agent.run(user_request=args.task)

    print(final_output)


if __name__ == "__main__":
    main()