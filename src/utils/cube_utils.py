import datetime
import os
import psycopg2

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferWindowMemory

# from council.chains import Chain
# from council.skills import LLMSkill
# from council.controllers import LLMController
# # from council.evaluators import LLMEvaluator
# from council.agents import Agent
# from council.filters import BasicFilter
# from council.llm import AzureLLM
# from council.contexts import ExecutionLog, ExecutionContext, AgentContext
# from council.contexts import ChatHistory, AgentContextStore, Budget

from typing import List, Optional

# from council.contexts import AgentContext, ChatMessage, ScoredChatMessage, ContextBase
# from council.evaluators import EvaluatorBase, EvaluatorException
# from council.llm import LLMBase, MonitoredLLM, llm_property, LLMAnswer, LLMMessage
# from council.llm.llm_answer import LLMParsingException
# from council.utils import Option

import dotenv

from langchain.chat_models import AzureChatOpenAI

memory = ConversationBufferWindowMemory(memory_key="chat_history", k=1)

_postgres_prompt = """\
You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query to run and return it as the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL. 
Never query for all columns from a table. You must query only the columns that are needed to answer the question.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Create meaningful aliases for the columns. For example, if the column name is products_sold.count, you should it as total_sold_products.
Note that the columns with (member_type: measure) are numeric columns and the ones with (member_type: dimension) are string columns.
You should include at least one column with (member_type: measure) in your query.
There are two types of queries supported against cube tables: aggregated and non-aggregated. Aggregated are those with GROUP BY statement, and non-aggregated are those without. Cube queries issued to your database will always be aggregated, and it doesn't matter if you provide GROUP BY in a query or not.
Whenever you use a non-aggregated query you need to provide only column names in SQL:

SELECT status, count FROM orders

The same aggregated query should always aggregate measure columns using a corresponding aggregating function or special MEASURE() function:

SELECT status, SUM(count) FROM orders GROUP BY 1
SELECT status, MEASURE(count) FROM orders GROUP BY 1

Strictly follow the instructions :
1. Always use date ranges for time related queries instead of extracting specific elements like month or year from the 'eventtimestamp' column. For Example, if the question is about total crane delay hours for a specific month or year, use a date range filter such as "eventtimestamp>='start_date' AND eventtimestamp < 'end_date'" rather than using functions like 'EXTRACT'.
2. Always Return the output in below json format :
{sql_query}


If you can't construct the query answer `{no_answer_text}`

Only use the following table: {table_info}

Only look among the following columns and pick the relevant ones: 


{columns_info}

user_question: {input_question}

- Use the chat history to understand the context of the user's question and provide a relevant SQL query. Ensure that the SQL query is directly relevant to the user's question, taking into account the provided chat history. Use the chat history to identify patterns or recurring themes in the user's questions, and tailor your SQL query accordingly.

chat history : {chat_history}

### Focus soly on constructing SQL QUERY without including any additional natural language or explanations in response ###

"""


PROMPT_POSTFIX = """\
Return the answer as a JSON object with the following format:

{
    "query": "",
    "filters": [{"column": \"\", "operator": \"\", "value": "\"\"}]
}
"""

Prompt_for_natural_lan_ans = """\
For the below provided user question and answer dataframe generate text answer.
Purely refer the given dataframe only including hearder for generating text answer assuming dataframe always gives you the answe to the user question
ONLY GENERATE ANSWER IN TEXT FORMAT FROM GIVEN ANSWER
BE SPECIFIC WITH SHORT ANSWER
DO NOT GENERATE EXTRA WORDING APART FROM ANWER REFERING TO DATAFRAME AND QUESTION.
Question: {input_question}
Dataframe: {dataframe}
"""

CUBE_SQL_API_PROMPT = PromptTemplate(
    input_variables=[
        "input_question",
        "table_info",
        "columns_info",
        "top_k",
        "no_answer_text",
        "chat_history",
        "sql_query"
    ],
    template=_postgres_prompt,
)

_NO_ANSWER_TEXT = "I can't answer this question."



def call_sql_api(sql_query: str):
    # load_dotenv()
    CONN_STR = os.environ["CUBE_DATABASE_URL"]
    print("CONN_STR: " + CONN_STR)

    # Initializing Cube SQL API connection)
    connection = psycopg2.connect(CONN_STR)

    cursor = connection.cursor()
    cursor.execute(sql_query)

    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    return columns, rows


def create_docs_from_values(columns_values, table_name, column_name):
    value_docs = []

    for column_value in columns_values:
        print(column_value)
        metadata = dict(
            table_name=table_name,
            column_name=column_name,
        )

        page_content = column_value
        value_docs.append(Document(page_content=page_content, metadata=metadata))

    return value_docs

def build_tools_text(tools: Sequence[ToolMetadata]) -> str:
    tools_dict = {}
    for tool in tools:
        tools_dict[tool.name] = tool.description
    return json.dumps(tools_dict, indent=4)


PREFIX = """\
        Given a user question for sql query generation as a ai assistance you have to split that query into 
        sub-queries for sql generation. Response of first query will be passed to consicutive query and used.
        Generate sub-queries accordingly so combining all queries will give answer to user question.
        GENERATE SUB_QUESTIONS WHEN YOU THINK THERE IS NEED OF SUB SQL QUERY IS REQUIRED FOR USER QUESTION BASED ON THE TABLE METADATA PROVIDED BELOW.
        GENERATE SUB-QUESTION ONLY WHEN YOU THINK THAT THERE IS A NECCESSITY OF GETTING ANSWER FOR FIRST QUERY IN ORDER TO ANSWER SECOND, ELSE RETURN THE USER QUESTION AS IT IN THE OUTPUT.

"""


example_query_str1 = (
    "What is the avg cmph for the recent visit of X-PRESS KAVERI?"
)
example_tools1 = [
    ToolMetadata(description={'table_name': 'crane_level_dim','columns_info': 'title: Crane Level Dim average executed crane moves per hour alias cmph, column name: crane_level_dim.total_executed_cmph, datatype: number, member type: measure\n\ntitle: Crane Level Dim Recent Visit, column name: crane_level_dim.recent_visit, datatype: number, member type: measure\n\ntitle: Crane Level Dim Executed Cmph, column name: crane_level_dim.executed_cmph, datatype: number, member type: dimension\n\ntitle: Crane Level Dim Executed Cmph Fm, column name: crane_level_dim.executed_cmph_fm, datatype: number, member type: measure\n\ntitle: Crane Level Dim total crane moves, column name: crane_level_dim.total_crane_moves_sub, datatype: number, member type: measure\n\ntitle: Crane Level Dim total crane delay hours, column name: crane_level_dim.total_crane_delay_hrs_sub, datatype: number, member type: measure\n\ntitle: Crane Level Dim total net crane moves hours, column name: crane_level_dim.net_crane_moves_hours_sub, datatype: number, member type: measure\n\ntitle: Crane Level Dim Last Crane Move Fm, column name: crane_level_dim.last_crane_move_fm, datatype: number, member type: measure\n\ntitle: Crane Level Dim Fm Composite Key, column name: crane_level_dim.fm_composite_key, datatype: string, member type: dimension\n\ntitle: Crane Level Dim Visitkey, column name: crane_level_dim.visitkey, datatype: string, member type: dimension\n\ntitle: Crane Level Dim Crane Moves Hours Fm, column name: crane_level_dim.crane_moves_hours_fm, datatype: number, member type: measure\n\ntitle: Crane Level Dim Crane Delay Hrs Fm, column name: crane_level_dim.crane_delay_hrs_fm, datatype: number, member type: measure\n\ntitle: Crane Level Dim Crane Delay Hrs, column name: crane_level_dim.crane_delay_hrs, datatype: number, member type: dimension\n\ntitle: Crane Level Dim Net Crane Moves Hours, column name: crane_level_dim.net_crane_moves_hours, datatype: number, member type: dimension\n\ntitle: Crane Level Dim Net Crane Moves Hours Fm, column name: crane_level_dim.net_crane_moves_hours_fm, datatype: number, member type: measure'}, 
                 name='crane_level_dim'),
]
example_tools_str1 = build_tools_text(example_tools1)
example_output1 = [
    SubQuestion(
        sub_question="What is the recent visit for X-PRESS KAVERI?", tool_name="crane_level_dim"
    ),
    SubQuestion(sub_question="What is the cmph for X-PRESS KAVERI on the recent visit?", tool_name="crane_level_dim"),
                ]
example_output_str1 = json.dumps({"items": [x.dict() for x in example_output1]}, indent=4)


example_query_str2 = (
    "What are the top 5 vessel having high cmph?"
)
example_tools2 = [
    ToolMetadata(description={'table_name': 'crane_level_dim', 'columns_info': 'title: Crane Level Dim average executed crane moves per hour alias cmph, column name: crane_level_dim.total_executed_cmph, datatype: number, member type: measure\n\ntitle: Crane Level Dim Vessel Id, column name: crane_level_dim.vessel_id, datatype: number, member type: dimension\n\ntitle: Crane Level Dim Vessel Name, column name: crane_level_dim.vessel_name, datatype: string, member type: dimension\n\ntitle: Crane Level Dim Executed Cmph, column name: crane_level_dim.executed_cmph, datatype: number, member type: dimension\n\ntitle: Crane Level Dim Executed Cmph Fm, column name: crane_level_dim.executed_cmph_fm, datatype: number, member type: measure\n\ntitle: Crane Level Dim total crane moves, column name: crane_level_dim.total_crane_moves_sub, datatype: number, member type: measure\n\ntitle: Crane Level Dim total crane delay hours, column name: crane_level_dim.total_crane_delay_hrs_sub, datatype: number, member type: measure\n\ntitle: Crane Level Dim Total Cranes, column name: crane_level_dim.total_cranes, datatype: number, member type: measure\n\ntitle: Crane Level Dim total net crane moves hours, column name: crane_level_dim.net_crane_moves_hours_sub, datatype: number, member type: measure\n\ntitle: Crane Level Dim Cheid, column name: crane_level_dim.cheid, datatype: string, member type: dimension\n\ntitle: Crane Level Dim Fm Composite Key, column name: crane_level_dim.fm_composite_key, datatype: string, member type: dimension\n\ntitle: Crane Level Dim Recent Visit, column name: crane_level_dim.recent_visit, datatype: number, member type: measure\n\ntitle: Crane Level Dim Crane Moves Fm, column name: crane_level_dim.crane_moves_fm, datatype: number, member type: measure\n\ntitle: Crane Level Dim Last Crane Move Fm, column name: crane_level_dim.last_crane_move_fm, datatype: number, member type: measure\n\ntitle: Crane Level Dim Crane Delay Hrs Fm, column name: crane_level_dim.crane_delay_hrs_fm, datatype: number, member type: measure'},
                  name='crane_level_dim'),
]
example_tools_str2 = build_tools_text(example_tools2)
example_output2 = [
    SubQuestion(
        sub_question="What are the top 5 vessel having high cmph?", tool_name="crane_level_dim"
    ),
                ]
example_output_str2 = json.dumps({"items": [x.dict() for x in example_output2]}, indent=4)


EXAMPLES = f"""\
# Example 1
<Tools>
```json
{example_tools_str1}
```

<User Question>
{example_query_str1}

<Output>
```json
{example_output_str1}

# Example 2
<Tools>
```json
{example_tools_str2}
```

<User Question>
{example_query_str2}

<Output>
```json
{example_output_str2}

""".replace(
    "{", "{{"
).replace(
    "}", "}}"
)

SUFFIX = """\
# Example 3
<Tools>
```json
{tools_str}
```

<User Question>
{query_str}

<Output>
"""

SUB_QUESTION_PROMPT_TMPL = PREFIX + EXAMPLES + SUFFIX

def get_subquestion(user_question, question_gen, vectorstore):

    docs = vectorstore.similarity_search(user_question, filter = dict(column_member_type='measure'))
    table_name = docs[0].metadata["table_name"]

    columns_question = user_question

    column_docs = vectorstore.similarity_search(
        columns_question, filter=dict(column_member_type='measure', table_name=table_name), k=15
    )

    lines = []
    for column_doc in column_docs:
        column_title = column_doc.metadata["column_title"]
        column_name = column_doc.metadata["column_name"]
        column_data_type = column_doc.metadata["column_data_type"]
        lines.append(
            f"title: {column_title}, column name: {column_name}, datatype: {column_data_type}, member type: {column_doc.metadata['column_member_type']}"
        )
    columns = "\n\n".join(lines)

    tools = [
                ToolMetadata(
                    name=docs[0].metadata['table_name'],
                    description= {'table_name':table_name, 'columns_info':columns},
                )
            ]
    
    sub_questions = question_gen.generate(
                                            tools=tools,
                                            query=QueryBundle(user_question),
                                            )
    return sub_questions


gq_prompt = """ You are responding to every prompt with general answer from the Metadata provided to you as below.
                Reposnd with "Sorry cannot answer your question." if the answer is not 
                present in metadata.

                "Metadata:"{conversational_metadata}

                "User Question" : {question}
            """


        

# class SpecialistGrade:
#     def __init__(self, index: int, grade: float, justification: str):
#         self._grade = grade
#         self._index = index
#         self._justification = justification

#     @llm_property
#     def grade(self) -> float:
#         """Your Grade"""
#         return self._grade

#     @llm_property
#     def index(self) -> int:
#         """Index of the answer graded in the list"""
#         return self._index

#     @llm_property
#     def justification(self) -> str:
#         """Short, helpful and specific explanation your grade"""
#         return self._justification

#     def __str__(self):
#         return f"Message {self._index} graded {self._grade} with the justification {self._justification}"


# class LLMEvaluator(EvaluatorBase):
#     """Evaluator using an `LLM` to evaluate chain responses."""

#     def __init__(self, llm: LLMBase):
#         """
#         Build a new LLMEvaluator.

#         :param llm: model to use for the evaluation.
#         """
#         super().__init__()
#         self._llm = self.register_monitor(MonitoredLLM("llm", llm))
#         self._llm_answer = LLMAnswer(SpecialistGrade)
#         self._retry = 3

#     def _execute(self, context: AgentContext) -> List[ScoredChatMessage]:
#         query = context.chat_history.try_last_user_message.unwrap()
#         chain_results = [
#             chain_messages.try_last_message.unwrap()
#             for chain_messages in context.chains
#             if chain_messages.try_last_message.is_some()
#         ]

#         retry = self._retry
#         messages = self._build_llm_messages(query, chain_results)
    
#         new_messages: List[LLMMessage] = []
#         while retry > 0:
#             retry -= 1
#             messages = messages + new_messages
#             llm_result = self._llm.post_chat_request(context, messages)
#             response = llm_result.first_choice
#             context.logger.debug(f"llm response: {response}")
#             try:
#                 parse_response = self._parse_response(context, response, chain_results)
#                 return parse_response
#             except LLMParsingException as e:
#                 assistant_message = f"Your response is not correctly formatted:\n{response}"
#                 new_messages = self._handle_error(e, assistant_message, context)
#             except EvaluatorException as e:
#                 assistant_message = f"Your response raised an exception:\n{response}"
#                 new_messages = self._handle_error(e, assistant_message, context)

#         raise EvaluatorException("LLMEvaluator failed to execute.")

#     @staticmethod
#     def _handle_error(e: Exception, assistant_message: str, context: ContextBase) -> List[LLMMessage]:
#         error = f"{e.__class__.__name__}: `{e}`"
#         context.logger.warning(f"Exception occurred: {error}")
#         return [LLMMessage.assistant_message(assistant_message), LLMMessage.user_message(f"Fix:\n{error}")]

#     def _parse_response(
#         self, context: ContextBase, response: str, chain_results: List[ChatMessage]
#     ) -> List[ScoredChatMessage]:
#         parsed = [self._parse_line(line) for line in response.strip().splitlines()]
#         grades = [r.unwrap() for r in parsed if r.is_some()]
#         if len(grades) == 0:
#             raise LLMParsingException("None of your grade could be parsed. Follow exactly formatting instructions.")

#         scored_messages = []
#         missing = []
#         for idx, message in enumerate(chain_results):
#             try:
#                 grade = next(filter(lambda item: item.index == (idx + 1), grades))
#                 scored_message = ScoredChatMessage(
#                     ChatMessage.agent(message=message.message, data=message.data), grade.grade
#                 )
#                 scored_messages.append(scored_message)
#                 context.logger.debug(f"{grade} {message.message}")
#             except StopIteration:
#                 missing.append(idx)

#         if len(missing) > 0:
#             raise EvaluatorException(f"Grade ALL {len(chain_results)} answers. Missing grade for {missing} answers.")

#         return scored_messages

#     def _build_llm_messages(self, query: ChatMessage, skill_messages: List[ChatMessage]) -> List[LLMMessage]:
#         if len(skill_messages) <= 0:
#             return []

#         responses = [skill_message.message for skill_message in skill_messages]
#         return [self._build_system_message(), self._build_user_message(query.message, responses)]

#     def _parse_line(self, line: str) -> Option[SpecialistGrade]:
#         if LLMAnswer.field_separator() not in line:
#             return Option.none()

#         cs: Optional[SpecialistGrade] = self._llm_answer.to_object(line)
#         return Option(cs)

#     @staticmethod
#     def _build_user_message(query: str, answers: list[str]) -> LLMMessage:
#         prompt_answers = "\n".join(
#             f"- answer #{index + 1} is: {answer if len(answer) > 0 else 'EMPTY'}"
#             for index, answer in enumerate(answers)
#         )
#         lines = [
#             "The question to grade is:",
#             query,
#             "Please grade the following answers according to your instructions:",
#             prompt_answers,
#         ]
#         prompt = "\n".join(lines)
#         return LLMMessage.user_message(prompt)

#     def _build_system_message(self) -> LLMMessage:
#         """Build prompt that will be sent to the inner `LLM`."""
#         task_description = [
#             "\n# ROLE",
#             "You are an instructor, with a large breadth of knowledge.",
#             "You are grading with objectivity answers from different Specialists to a given question.",
#             "\n# INSTRUCTIONS",
#             "1. Give a grade from 0.0 to 10.0",
#             "2. Evaluate carefully the question and the proposed answer.",
#             "3. Ignore how assertive the answer is, only content accuracy count for grading."
#             "4. Consider only the Specialist's answer and ignore its index for grading.",
#             "5. Ensure to be consistent in grading, identical answers must have the same grade.",
#             "6. Irrelevant, inaccurate, inappropriate, false or empty answer must be graded 0.0",
#             "7. Agent should never mix-up the response with SQL query and general answer if thats the case answer must be graded 0.0. Example: SELECT total_executed_cmph AS total_executed_cmph_metric_explanation FROM crane_level_dim\n\nThis query will return the metric 'total executed cmph' from the 'crane_level_dim' table. The metric is represented by the column 'total_executed_cmph' and is of member type 'measure'.",
#             "8. First check the user question intent break it down into thoughts that wheather user want question to be answerd in natural language format or SQL query. Example: {'question':'What is the crane count and how it can be used to calculate efficiency?', 'Thought':' If I break the question first part 'What is the crane count' imply user wants count of cranes from SQL query but the second part 'how it can be used to calculate efficiency?' here user wants the general explaination so answers should be in general natural language format. So, I'll be grading the answers with high score having natural language response.}",
#             ""
#             "\n# FORMATTING",
#             "1. The list of given answers is formatted precisely as:",
#             "- answer #{index} is: {Specialist's answer or EMPTY if no answer}",
#             "2. For each given answer, format your response precisely as:",
#             self._llm_answer.to_prompt(),
#         ]
#         prompt = "\n".join(task_description)
#         return LLMMessage.system_message(prompt)

# class CouncilAI(LLMEvaluator):
    
#     def __init__(self,
#                  conversational_metadata = list,
#                  question = str,
#                  sq_prompt = str,
#                  gq_prompt = str):
        
#         self.conversational_metadata = conversational_metadata
#         self.question = question
#         self.sq_prompt = sq_prompt
#         self.gq_prompt = gq_prompt
        
#     def council_agent(self):

#         dotenv.load_dotenv()
#         # Load OpenAILLM
#         openai_llm = AzureLLM.from_env()

#         sq_skill = LLMSkill(llm=openai_llm, system_prompt=self.sq_prompt)
#         sq_chain = Chain(name="SQL Agent", description=""" This skill can be used to answer the user question related to the metrics that can be calculated on the SQL database.
#                          Use this skill to answer the user questions/input query similar to below:
#                          1. What is the average cmph for the recent visit of vessel X-PRESS KAVERI?
#                          2. What is cmph for last weeek?
#                          3. What is the crane name that made less number of moves?
#                          \n# INSTRUCTIONS #
#                          1. FOR THIS CHAIN IF THERE IS ANY OTHER TEXT PRESENT IN RESPONSE OTHER THAN SQL QUERY STRICTLY ASSIGN SCORE TO 0.0
#                          2. These skill can answers to the questions for calculating metrics with time range like some dates, days or date range.
#                          3. It can answers the questions for finding entities like timesstamps, vessel names, terminal names, cheid's etc on database.""",
#                          runners=[sq_skill])
        
#         gq_skill = LLMSkill(llm=openai_llm, system_prompt=self.gq_prompt)
#         gq_chain = Chain(name="GQ Agent", description=""" This skill can be used to the general user questions related to explaination, description, interpretation of metrics, tables or columns from provided metadata.
#                          use this skill to answer the user quetions/input queries as below:
#                          1. Explain me the metric CMPH.
#                          2. What dose the vessel name column represents?
#                          3. What is the metric that is related to crane moves?
                         
#                          ### INSTRUCTIONS ###
#                          1.For the user question where keyworrds like 'defination','Explain','Represents','Tell me','Describe','Interprete' are there for those questions give more score to GQ agent."
#                          """,
#                         runners=[gq_skill])
        
#         controller = LLMController(llm=openai_llm, chains=[sq_chain, gq_chain], response_threshold=5)

#         evaluator = LLMEvaluator(llm=openai_llm)

#         chat_history = ChatHistory()
#         chat_history.add_user_message(self.question)
#         execution_log = ExecutionLog()

#         agent_context_store = AgentContextStore(chat_history)
#         execution_context = ExecutionContext(execution_log)
#         budget = Budget(600)

#         run_context = AgentContext(agent_context_store, execution_context, budget )

#         agent = Agent(controller=controller, evaluator=evaluator, filter=BasicFilter())

#         response = agent.execute(run_context)

#         answers = []
#         dict_log = execution_log.to_dict()

#         for entry in dict_log['entries']:
#             try:
#                 if entry['messages'][0]['source'] != "" and str(entry['source']).split("/")[-1]=="sequence[1]":
#                     source = str(entry['source']).split("/")[2].split("(")[1].replace(")","")
#                     message = entry['messages'][0]['message']
#                     score = entry['source']
#                     answers.append({"Message":message,"Source":source})
#                 elif entry['messages'][0]['source'] != "" and str(entry['source']).split("/")[-1]=="runner":
#                     source = str(entry['source']).split("/")[2].split("(")[1].replace(")","")
#                     message = entry['messages'][0]['message']
#                     score = entry['source']
#                     answers.append({"Message":message,"Source":source})
#             except:
#                         print("no entry")


#         source = [answers[i]['Source'] for i in range(len(answers)) if answers[i]['Message'] == response.best_message.message]
        
#         agent_result_answers = {}
#         for idx, msg in enumerate(response.messages):
#             agent_result_answers["message_"+str(idx)] = msg.message._message
#             agent_result_answers["score_"+str(idx)] = msg.score

#         return response.best_message.message, source, agent_result_answers
