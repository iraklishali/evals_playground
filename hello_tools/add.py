from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import (
    generate,
    use_tools,
)
from inspect_ai.tool import tool

@tool
def add():
    async def execute(x: int, y: int):
        """
        Add two numbers.

        Args:
            x: First number to add.
            y: Second number to add.

        Returns:
            The sum of the two numbers.
        """
        return x + y

    return execute


@task
def addition_problem():
    return Task(
        dataset=[Sample(input="What is 1 + 1?", target=["2", "2.0"])],
        solver=[use_tools(add()), generate()],
        scorer=match(numeric=True),
    )

@task
def addition_problem_hard():
    return Task(
        dataset=[Sample(input="What is 123444565641 + 298435629337?", target=["421880194978", "421880194978.0"])],
        solver=[use_tools(add()), generate()],
        scorer=match(numeric=True),
    )

@task
def addition_in_container():
    return Task(
        dataset=[Sample(input="What is 123444565641 + 298435629337?", target=["421880194978", "421880194978.0"])],
        solver=[use_tools(add()), generate()],
        sandbox="docker",
        scorer=match(numeric=True),
    )