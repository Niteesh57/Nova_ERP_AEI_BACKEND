import os
import traceback
from dotenv import load_dotenv

load_dotenv()

try:
    from nova_act import NovaAct, workflow
    print("NovaAct import successful.")

    # Setup explicit AWS credentials for Nova Act
    boto_config = {
        "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    }
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        boto_config["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID")
        boto_config["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if os.environ.get("AWS_SESSION_TOKEN"):
            boto_config["aws_session_token"] = os.environ.get("AWS_SESSION_TOKEN")

    print(f"Boto config constructed with region: {boto_config.get('region_name')}")
    print(f"Has access key: {'aws_access_key_id' in boto_config}")

    @workflow(
        workflow_definition_name="AutonomousMarketResearchAgent", 
        model_id="nova-act-latest",
        boto_session_kwargs=boto_config
    )
    def _run_nova_act_workflow(prompt_action: str):
        with NovaAct(starting_page="https://www.google.com") as nova:
            return nova.act(prompt_action)

    print("Executing NovaAct workflow...")
    result = _run_nova_act_workflow("Search Google for 'Testing'")
    print("Result:", result)
except ImportError as e:
    print("CAUGHT IMPORT ERROR:")
    traceback.print_exc()
except Exception as e:
    print("CAUGHT EXCEPTION:")
    traceback.print_exc()
