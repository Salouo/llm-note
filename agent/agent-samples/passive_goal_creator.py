from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# Set up environment variables
loaded = load_dotenv()

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


def main():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    goal_creator = PassiveGoalCreator(llm=llm)

    task = "明日は友達と遊びに行く！"
    result: Goal = goal_creator.run(query=task)

    print(f"{result.text}")


if __name__ == "__main__":
    main()