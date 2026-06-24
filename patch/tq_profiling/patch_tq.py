# Copyright (c) 2025 verl-project authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from functools import wraps
import inspect
import transfer_queue as tq
import rl_insight

def tq_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)

        cost = time.time() - start_time
        rl_insight.metric_value(f"TQ_TIME_{func.__name__}", float(cost))
        return result

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        await_result = await func(*args, **kwargs)

        cost = time.time() - start_time
        rl_insight.metric_value(f"TQ_TIME_{func.__name__}", float(cost))
        return await_result
    return async_wrapper if inspect.iscoroutinefunction(func) else wrapper

def patch_tq():
    tq_funcs = [
        "kv_batch_put",
        "kv_batch_get",
        "kv_clear",
        "async_kv_batch_put",
    ]
    rl_insight.init()
    for name in tq_funcs:
        if hasattr(tq, name):
            original = getattr(tq, name)
            setattr(tq, name, tq_timer(original))

def patch_tq_timer(cls):
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        patch_tq()
    cls.__init__ = new_init

    return cls

