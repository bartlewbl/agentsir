"""A small Lambda tool function to copy.

Deploy it with the handler set to `lambdas.example.handler`. The README has
the commands.
"""

import os

from lambdas.handler import make_handler


def lambda_status(context: dict) -> str:
    """Report which Lambda function answered. Use it to check that Lambda tools work."""
    function = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "a local run")
    region = os.environ.get("AWS_REGION", "no region")
    user = context.get("user_id") or "unknown"
    return f"Answered by {function} in {region} for user {user}."


handler = make_handler([lambda_status])
