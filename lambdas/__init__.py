"""Run agent tools in AWS Lambda functions.

`protocol.py` is the JSON both sides speak. `handler.py` is the Lambda side:
it turns plain functions into a Lambda handler. `client.py` and
`agent_tools.py` are the agent side: they call a function through boto3 and
hand its tools to the agent.

A Lambda zip needs this file, `protocol.py`, `handler.py`, and the module with
the tool functions. Those use the standard library only. `example.py` is one to copy.
"""
