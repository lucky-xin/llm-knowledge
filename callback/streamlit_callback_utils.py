import inspect
from typing import Callable, TypeVar

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx


def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    fn_return_type = TypeVar('fn_return_type')

    def add_streamlit_context(fn: Callable[..., fn_return_type]) -> Callable[..., fn_return_type]:
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> fn_return_type:
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)

        return wrapper

    st_cb = StreamlitCallbackHandler(
        max_thought_containers=2,
        parent_container=parent_container,
        collapse_completed_thoughts=True,
        expand_new_thoughts=False,
    )

    for method_name, method_func in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if method_name.startswith('on_'):
            setattr(st_cb, method_name, add_streamlit_context(method_func))
    return st_cb
