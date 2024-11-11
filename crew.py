from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from rag_tool import rag_agent_tool

# Uncomment the following line to use an example of a custom tool
# from ai_news.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from langchain_openai import ChatOpenAI

defaul_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5
)


@CrewBase
class InternalSupportCrew():
	"""Internal Support crew"""

	@agent
	def l1_support_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['l1_support_agent'],
			tools=[rag_agent_tool], # Example of custom tool, loaded on the beginning of file
			# tools=[MyCustomTool()], # Example of custom tool, loaded on the beginning of file
			verbose=True,
			model=defaul_llm,
			memory=False,
			max_iter=15,
			respect_context_window=True
		)

	@agent
	def responder_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['responder_agent'],
			verbose=True,
			model=defaul_llm,
			memory=False,
			allow_delegation=False
		)

	@task
	def l1_support_agent_task(self) -> Task:
		return Task(
			config=self.tasks_config['l1_support_agent_task'],
			output_file='report_intermediate.md'
		)

	@task
	def responder_agent_task(self) -> Task:
		return Task(
			config=self.tasks_config['responder_agent_task'],
			output_file='report_final7.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the l1 support crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			verbose=True,
			process=Process.sequential, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
			#manager_llm=defaul_llm,
			cache=False
		)