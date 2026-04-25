import autogen
import os
import json
from bias_engine import run_bias_pipeline

def dummy_fetch_data() -> str:
    """Invokes the data generation script to mock an API extract."""
    # We just ensure the file is generated
    import os
    if not os.path.exists("synthetic_pmfby_data.csv"):
        import generate_data
    return "SUCCESS: synthetic_pmfby_data.csv has been prepared."

def dummy_run_bias_pipeline() -> str:
    """Executes the machine learning fairness pipeline."""
    res = run_bias_pipeline("synthetic_pmfby_data.csv")
    return json.dumps(res)

def get_agent_system():
    # In a real environment, load from st.secrets or os.environ
    api_key = os.environ.get("OPENAI_API_KEY", "sk-mock-key-for-local-demo")
    
    llm_config = {
        "config_list": [{"model": "gpt-4o", "api_key": api_key}],
        "temperature": 0.2
    }

    data_agent = autogen.AssistantAgent(
        name="DataAgent",
        system_message="""You are the DataAgent. Your job is to fetch and clean the PMFBY agriculture data.
Call the `fetch_data` function. Once it returns success, output 'DATA_READY'.""",
        llm_config=llm_config,
    )

    bias_agent = autogen.AssistantAgent(
        name="BiasDetectionAgent",
        system_message="""You are the BiasDetectionAgent. Your job is to run the fairness metrics pipeline.
Call `run_pipeline`. Review the Baseline vs Mitigated Equity Scores.
FLAG proxy variables if any feature contribution is >15% in SHAP (e.g. Irrigation_Type).
Summarize findings and output 'BIAS_ANALYSIS_COMPLETE'.""",
        llm_config=llm_config,
    )

    report_agent = autogen.AssistantAgent(
        name="ReportAgent",
        system_message="""You are the ReportAgent. You synthesize technical bias metrics into an Equity Audit Report.
Draft a Markdown report. 
Include this exact policy recommendation: "Ministry of Agriculture should mandate quarterly bias audits on all AI-assisted PMFBY claim processing systems."
Terminate your message with 'REPORT_FINALIZED'.""",
        llm_config=llm_config,
    )

    manager = autogen.AssistantAgent(
        name="ManagerAgent",
        system_message="""You are the ManagerAgent. Orchestrate the pipeline exactly as follows:
1. First, tell DataAgent to fetch data.
2. Wait for DataAgent to say 'DATA_READY'. Then, tell BiasDetectionAgent to run the pipeline.
3. Wait for BiasDetectionAgent to say 'BIAS_ANALYSIS_COMPLETE'. Then, tell ReportAgent to write the report.
4. Wait for ReportAgent to say 'REPORT_FINALIZED'. Then output 'TERMINATE'.""",
        llm_config=llm_config,
    )

    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", ""),
        code_execution_config={"use_docker": False}
    )

    # Register functions
    autogen.agentchat.register_function(
        dummy_fetch_data, caller=data_agent, executor=user_proxy, name="fetch_data", description="Fetches PMFBY data"
    )
    autogen.agentchat.register_function(
        dummy_run_bias_pipeline, caller=bias_agent, executor=user_proxy, name="run_pipeline", description="Runs bias metrics"
    )

    groupchat = autogen.GroupChat(
        agents=[user_proxy, manager, data_agent, bias_agent, report_agent], 
        messages=[], 
        max_round=15,
        speaker_selection_method="round_robin",
        allow_repeat_speaker=False
    )
    
    manager_chat = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    return user_proxy, manager_chat

def run_agents():
    user, manager_chat = get_agent_system()
    user.initiate_chat(manager_chat, message="Start the PMFBY unbiased data audit pipeline.")

if __name__ == "__main__":
    run_agents()
