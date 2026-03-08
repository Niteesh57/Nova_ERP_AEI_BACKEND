import traceback
from nova_act import NovaAct, workflow

try:
    @workflow(workflow_definition_name="AutonomousMarketResearchAgent", model_id="nova-act-latest")
    def _run_nova_act_workflow(prompt_action: str):
        with NovaAct(starting_page="https://www.google.com") as nova:
            return nova.act(prompt_action)

    print("Attempting to run NovaAct...")
    result = _run_nova_act_workflow("Search Google for 'Testing'")
    print("SUCCESS Result:", result)
except Exception as e:
    print("FAILED with exception:")
    traceback.print_exc()
