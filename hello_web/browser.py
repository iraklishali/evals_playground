from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import web_browser


@task
def browser():
    return Task(
        dataset=[
            Sample(
                input="Where and When was Sergei Parajanov born? Feel free to look up in the internet."
            )
        ],
        solver=[
            use_tools(web_browser()),
            generate(),
        ],
        scorer=includes(),
        sandbox="docker",
    )