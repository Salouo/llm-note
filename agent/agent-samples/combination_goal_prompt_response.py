from passive_goal_creator import PassiveGoalCreator, Goal
from prompt_optimizer import PromptOptimizer, OptimizedGoal
from response_optimizer import ResponseOptimizer
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


# Set up environment variables
loaded = load_dotenv()


def main():
    # Instantiate a llm
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Instantiate agents
    passive_goal_creator = PassiveGoalCreator(llm=llm)
    prompt_optimizer = PromptOptimizer(llm=llm)
    response_optimizer = ResponseOptimizer(llm=llm)

    # Define the task
    task = "カレーライスの作り方"

    # Combination
    goal: Goal = passive_goal_creator.run(query=task)
    optimized_goal: OptimizedGoal = prompt_optimizer.run(query=goal.text)
    optimized_response: str = response_optimizer.run(query=optimized_goal.text)

    # Print the result
    print(f"{optimized_response}")


if __name__ == "__main__":
    main()
    