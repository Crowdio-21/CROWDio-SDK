"""
AST-based code instrumentation for mobile workers

This module provides code transformation for workers that don't support
sys.settrace() (e.g., Chaquopy on Android). It uses AST manipulation to
inject explicit checkpoint calls into function code.

Key Features:
- CheckpointInstrumenter: Injects _ckpt_update() calls at strategic points
- ResumeInstrumenter: Transforms code to resume from checkpoint state
- Mobile checkpoint wrapper: Provides explicit state capture for mobile

This is TRANSPARENT to developers - they write pure logic, and the framework
automatically instruments code for mobile workers.
"""

import ast
import enum
import logging
from typing import List, Dict, Any, Optional, Tuple


logger = logging.getLogger(__name__)


class WorkerType(enum.Enum):
    """Types of workers in the system"""
    PC_PYTHON = "pc_python"           # Standard Python with sys.settrace() support
    ANDROID_CHAQUOPY = "android_chaquopy"  # Chaquopy - no settrace
    ANDROID_KOTLIN = "android_kotlin"      # Kotlin-based worker
    UNKNOWN = "unknown"


def detect_worker_capabilities(
    platform: str = None,
    runtime: str = None,
    capabilities: Dict[str, bool] = None
) -> Dict[str, Any]:
    """
    Detect worker capabilities based on platform info
    
    Args:
        platform: Platform identifier (e.g., "android", "windows", "linux")
        runtime: Runtime identifier (e.g., "chaquopy", "cpython")
        capabilities: Optional pre-defined capabilities dict
        
    Returns:
        Dictionary of capabilities including:
        - supports_settrace: Whether sys.settrace() works
        - supports_frame_introspection: Whether frame locals work
        - worker_type: WorkerType enum value
    """
    result = {
        "supports_settrace": True,
        "supports_frame_introspection": True,
        "worker_type": WorkerType.PC_PYTHON,
        "needs_instrumentation": False
    }
    
    # Check explicit capabilities first
    if capabilities:
        if capabilities.get("supports_settrace") is False:
            result["supports_settrace"] = False
            result["needs_instrumentation"] = True
        if capabilities.get("supports_frame_introspection") is False:
            result["supports_frame_introspection"] = False
            result["needs_instrumentation"] = True
    
    # Detect based on platform/runtime
    if platform and platform.lower() == "android":
        if runtime and "chaquopy" in runtime.lower():
            result["worker_type"] = WorkerType.ANDROID_CHAQUOPY
            result["supports_settrace"] = False
            result["supports_frame_introspection"] = False
            result["needs_instrumentation"] = True
        elif runtime and "kotlin" in runtime.lower():
            result["worker_type"] = WorkerType.ANDROID_KOTLIN
            result["supports_settrace"] = False
            result["supports_frame_introspection"] = False
            result["needs_instrumentation"] = True
    
    return result


class CheckpointInstrumenter(ast.NodeTransformer):
    """
    AST transformer that injects checkpoint update calls into code
    
    Transforms:
    - Loop bodies: Injects _ckpt_update() at end of each iteration
    - Key assignments: Captures state after checkpoint variables are updated
    
    Example transformation:
    
    Before:
        for i in range(n):
            total += compute(data[i])
            progress_percent = (i + 1) / n * 100
    
    After:
        for i in range(n):
            total += compute(data[i])
            progress_percent = (i + 1) / n * 100
            _ckpt_update({'i': i, 'total': total, 'progress_percent': progress_percent})
    """
    
    def __init__(self, checkpoint_state_vars: List[str]):
        """
        Initialize instrumenter
        
        Args:
            checkpoint_state_vars: List of variable names to capture
        """
        super().__init__()
        self.checkpoint_state_vars = checkpoint_state_vars
        self.loops_instrumented = 0
        self.assignments_instrumented = 0

    @staticmethod
    def _is_pause_wait_test(test: ast.expr) -> bool:
        """Return True for task-control pause wait guards."""
        if isinstance(test, ast.Name):
            return test.id == 'paused'

        if isinstance(test, ast.Attribute):
            return (
                isinstance(test.value, ast.Name)
                and test.value.id == '_TaskControl'
                and test.attr == 'paused'
            )

        if isinstance(test, ast.Call):
            if isinstance(test.func, ast.Attribute):
                return (
                    isinstance(test.func.value, ast.Name)
                    and test.func.value.id == '_TaskControl'
                    and test.func.attr == 'is_paused'
                )

        return False
    
    def _create_ckpt_update_call(self) -> ast.Try:
        """
        Create AST for: try: _ckpt_update({var1: var1, ...}); except NameError: pass
        """
        keys = [ast.Constant(value=var) for var in self.checkpoint_state_vars]
        values = [ast.Name(id=var, ctx=ast.Load()) for var in self.checkpoint_state_vars]
        state_dict = ast.Dict(keys=keys, values=values)
        call = ast.Call(
            func=ast.Name(id='_ckpt_update', ctx=ast.Load()),
            args=[state_dict],
            keywords=[]
        )
        return ast.Try(
            body=[ast.Expr(value=call)],
            handlers=[ast.ExceptHandler(
                type=ast.Name(id='NameError', ctx=ast.Load()),
                name=None,
                body=[ast.Pass()]
            )],
            orelse=[],
            finalbody=[]
        )
    
    def visit_For(self, node: ast.For) -> ast.For:
        """Inject checkpoint call at end of for loop body, skip if already present."""
        self.generic_visit(node)
        # Idempotency: skip if _ckpt_update already present
        if not any(self._is_ckpt_update_call(stmt) for stmt in node.body):
            ckpt_call = self._create_ckpt_update_call()
            node.body.append(ckpt_call)
            self.loops_instrumented += 1
        return node
    
    def visit_While(self, node: ast.While) -> ast.While:
        """Inject checkpoint call at end of while loop body, skip if already present."""
        if self._is_pause_wait_test(node.test):
            return node
        self.generic_visit(node)
        if not any(self._is_ckpt_update_call(stmt) for stmt in node.body):
            ckpt_call = self._create_ckpt_update_call()
            node.body.append(ckpt_call)
            self.loops_instrumented += 1
        return node

    def visit_Assign(self, node: ast.Assign) -> Any:
        """Instrument assignments to checkpointed variables with checkpoint call."""
        self.generic_visit(node)
        # Only instrument if assignment is to a tracked variable
        if any(isinstance(t, ast.Name) and t.id in self.checkpoint_state_vars for t in node.targets):
            self.assignments_instrumented += 1
            return [node, self._create_ckpt_update_call()]
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return self.visit_FunctionDef(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        return self.visit_For(node)

    def _is_ckpt_update_call(self, node: ast.stmt) -> bool:
        # Detects if a statement is a _ckpt_update call
        if isinstance(node, ast.Try):
            for stmt in node.body:
                if self._is_ckpt_update_call(stmt):
                    return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            return isinstance(func, ast.Name) and func.id == '_ckpt_update'
        return False


class TaskControlInstrumenter(ast.NodeTransformer):
    """
    AST transformer that injects pause/kill control checks into task code
    
        For Android workers where the UI needs to suspend, kill, or resume a task,
        this transformer injects:
    - Kill check (`if killed: return "killed"`) and pause wait
      (`while paused: time.sleep(0.1)`) at the start of loop bodies
    
    Example transformation:
    
    Before:
        def example_task(n):
            result = 0
            for i in range(n):
                result += i * i
            return result
    
    After:
        def example_task(n):
            result = 0
            for i in range(n):
                if _TaskControl.is_killed():
                    return 'killed'
                while _TaskControl.is_paused():
                    time.sleep(0.1)
                result += i * i
            return result
    """
    
    def __init__(self, checkpoint_state_vars: Optional[List[str]] = None):
        super().__init__()
        self.checkpoint_state_vars = checkpoint_state_vars or []
        self.functions_instrumented = 0
        self.loops_instrumented = 0
    
    def _create_kill_check(self) -> ast.If:
        """Create AST for: if _TaskControl.is_killed(): return 'killed'"""
        body = [ast.Return(value=ast.Constant(value='killed'))]
        return ast.If(
            test=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='_TaskControl', ctx=ast.Load()),
                    attr='is_killed',
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            ),
            body=body,
            orelse=[]
        )
    
    def _create_pause_wait(self) -> ast.While:
        """Create AST for: while _TaskControl.is_paused(): time.sleep(0.1)"""
        return ast.While(
            test=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='_TaskControl', ctx=ast.Load()),
                    attr='is_paused',
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            ),
            body=[
                ast.Expr(value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='time', ctx=ast.Load()),
                        attr='sleep',
                        ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=0.1)],
                    keywords=[]
                ))
            ],
            orelse=[]
        )
    
    def _create_control_checks(self) -> List[ast.stmt]:
        """Create the kill check + pause wait pair"""
        return [self._create_kill_check(), self._create_pause_wait()]
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Track function instrumentation count"""
        # First, visit children (to instrument nested loops)
        self.generic_visit(node)
        self.functions_instrumented += 1

        return node
    
    def visit_For(self, node: ast.For) -> ast.For:
        """Inject kill check and pause wait at start of for loop body"""
        # First, visit children
        self.generic_visit(node)
        
        # Insert control checks at the start of the loop body
        control_stmts = self._create_control_checks()
        for i, stmt in enumerate(control_stmts):
            node.body.insert(i, stmt)
        self.loops_instrumented += 1
        
        return node
    
    def visit_While(self, node: ast.While) -> ast.While:
        """Inject kill check and pause wait at start of while loop body"""
        # Skip the `while paused:` loops we ourselves inject
        if (
            isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Attribute)
            and isinstance(node.test.func.value, ast.Name)
            and node.test.func.value.id == '_TaskControl'
            and node.test.func.attr == 'is_paused'
        ):
            return node
        
        # First, visit children
        self.generic_visit(node)
        
        # Insert control checks at the start of the loop body
        control_stmts = self._create_control_checks()
        for i, stmt in enumerate(control_stmts):
            node.body.insert(i, stmt)
        self.loops_instrumented += 1
        
        return node


class ResumeInstrumenter(ast.NodeTransformer):
    """
    AST transformer for checkpoint resume
    
    Transforms code to:
    1. Replace initial variable assignments with checkpoint values
    2. Adjust loop ranges to start from checkpoint position
    """
    
    def __init__(self, checkpoint_state: Dict[str, Any], checkpoint_state_vars: List[str]):
        """
        Initialize resume instrumenter
        
        Args:
            checkpoint_state: Dictionary with saved checkpoint state
            checkpoint_state_vars: List of variables to restore
        """
        super().__init__()
        self.checkpoint_state = checkpoint_state
        self.checkpoint_state_vars = checkpoint_state_vars
        self.modified_vars = set()
        self.found_main_loop = False
        self.in_function = False
        self.func_name = None
    
    def _create_ast_value(self, value: Any) -> Optional[ast.expr]:
        """Create an AST node for a Python value (no double call, correct None check)"""
        if isinstance(value, (int, float)):
            return ast.Constant(value=value)
        elif isinstance(value, str):
            return ast.Constant(value=value)
        elif isinstance(value, bool):
            return ast.Constant(value=value)
        elif value is None:
            return ast.Constant(value=None)
        elif isinstance(value, list):
            elts = []
            for v in value:
                node = self._create_ast_value(v)
                if node is not None:
                    elts.append(node)
            return ast.List(elts=elts, ctx=ast.Load())
        elif isinstance(value, dict):
            keys = []
            vals = []
            for k, v in value.items():
                key_node = ast.Constant(value=k)
                val_node = self._create_ast_value(v)
                if val_node is not None:
                    keys.append(key_node)
                    vals.append(val_node)
            return ast.Dict(keys=keys, values=vals)
        elif isinstance(value, tuple):
            elts = []
            for v in value:
                node = self._create_ast_value(v)
                if node is not None:
                    elts.append(node)
            return ast.Tuple(elts=elts, ctx=ast.Load())
        return None
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Track function entry"""
        self.in_function = True
        self.func_name = node.name
        self.generic_visit(node)
        self.in_function = False
        return node
    
    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Replace initial variable assignments with checkpoint values"""
        if not self.in_function:
            return node
        # Check if this is a simple assignment to a checkpointed variable
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            if var_name in self.checkpoint_state and var_name not in self.modified_vars:
                value = self.checkpoint_state[var_name]
                new_value = self._create_ast_value(value)
                if new_value is not None:
                    node.value = new_value
                    self.modified_vars.add(var_name)
        return node
    
    def visit_For(self, node: ast.For) -> ast.For:
        """Adjust for loop range to start from checkpointed loop index (not percent progress)"""
        if not self.in_function:
            return node
        # Only handle simple for i in range(n) or range(start, n)
        if isinstance(node.target, ast.Name):
            loop_var = node.target.id
            if loop_var in self.checkpoint_state:
                resume_from = self.checkpoint_state[loop_var]
                args = node.iter.args if isinstance(node.iter, ast.Call) else []
                if len(args) == 1:
                    # range(n) -> range(resume_from, n)
                    node.iter.args = [ast.Constant(value=resume_from), args[0]]
                elif len(args) >= 2:
                    # range(start, end) -> range(max(start, resume_from), end)
                    orig_start, end_arg = args[0], args[1]
                    start_expr = ast.Call(
                            func=ast.Name(id='max', ctx=ast.Load()),
                            args=[
                            orig_start,
                            ast.Constant(value=resume_from)
                        ],
                        keywords=[]
                    )
                    node.iter.args = [start_expr, args[1]] + args[2:]
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return self.visit_FunctionDef(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        return self.visit_For(node)


def instrument_for_mobile(
    func_code: str,
    checkpoint_state_vars: List[str]
) -> Tuple[str, int]:
    """
    Instrument function code for mobile workers without sys.settrace()
    
    Injects explicit _ckpt_update() calls at strategic points so the worker
    can capture state without frame introspection.
    
    Args:
        func_code: Original function source code
        checkpoint_state_vars: List of variable names to capture
        
    Returns:
        Tuple of (instrumented_code, num_loops_instrumented)
    """
    try:
        # Parse the code
        tree = ast.parse(func_code)
        
        # Apply instrumentation
        instrumenter = CheckpointInstrumenter(checkpoint_state_vars)
        tree = instrumenter.visit(tree)
        
        # Fix missing line numbers
        ast.fix_missing_locations(tree)
        
        # Convert back to code
        instrumented_code = ast.unparse(tree)
        
        return instrumented_code, instrumenter.loops_instrumented
        
    except Exception as e:
        logger.warning("[CodeInstrumenter] Instrumentation failed: %s", e)
        return func_code, 0


def prepare_code_for_mobile_resume(
    func_code: str,
    checkpoint_state: Dict[str, Any],
    checkpoint_state_vars: List[str]
) -> str:
    """
    Transform function code to resume from checkpoint on mobile worker
    
    Applies:
    1. Variable injection (replace initial values with checkpoint)
    2. Loop adjustment (start from checkpoint position)
    3. Instrumentation (inject _ckpt_update calls)
    
    Args:
        func_code: Original function source code
        checkpoint_state: Saved checkpoint state
        checkpoint_state_vars: List of variable names
        
    Returns:
        Transformed code ready for resumed execution
    """
    try:
        # Parse the code
        tree = ast.parse(func_code)
        
        # Apply resume transformation
        resume_instrumenter = ResumeInstrumenter(checkpoint_state, checkpoint_state_vars)
        tree = resume_instrumenter.visit(tree)
        
        # Apply checkpoint instrumentation
        checkpoint_instrumenter = CheckpointInstrumenter(checkpoint_state_vars)
        tree = checkpoint_instrumenter.visit(tree)
        
        # Fix missing line numbers
        ast.fix_missing_locations(tree)
        
        # Convert back to code
        return ast.unparse(tree)
        
    except Exception as e:
        logger.warning("[CodeInstrumenter] Resume transformation failed: %s", e)
        return func_code


def generate_mobile_checkpoint_wrapper() -> str:
    """
    Generate the checkpoint wrapper code that must be prepended to instrumented functions
    
    This provides:
    - _MobileCheckpointState: Thread-safe state container
    - _ckpt_update(): Function called by instrumented code
    - _ckpt_get_state(): Function for checkpoint handler to read state
    
    Returns:
        Python code string to prepend to instrumented functions
    """
    return '''
# Mobile checkpoint wrapper - provides explicit state capture
import threading

class _MobileCheckpointState:
    """Thread-safe checkpoint state container for mobile workers"""
    _lock = threading.Lock()
    _state = {}
    
    @classmethod
    def update(cls, state_dict):
        with cls._lock:
            cls._state.update(state_dict)
    
    @classmethod
    def get(cls):
        with cls._lock:
            return dict(cls._state)
    
    @classmethod
    def reset(cls):
        with cls._lock:
            cls._state = {}

def _ckpt_update(state_dict):
    """Called by instrumented code to update checkpoint state"""
    _MobileCheckpointState.update(state_dict)

def _ckpt_get_state():
    """Called by checkpoint handler to read current state"""
    return _MobileCheckpointState.get()
'''


def generate_task_control_wrapper() -> str:
    """
    Generate the task control wrapper code that must be prepended to instrumented functions
    
    This provides:
    - _TaskControl (thread-local paused/killed flags)
    - pause(), resume(), kill() control functions
    
    These flags are checked by the control checks injected by TaskControlInstrumenter.
    
    Returns:
        Python code string to prepend to instrumented functions
    """
    return '''# CROWDio runtime marker: task_control_wrapper_v1
__CROWDIO_TASK_CONTROL_WRAPPER__ = True

import time
import threading

class _TaskControl:
    """Thread-local task control flags for pause/kill."""
    _local = threading.local()

    @classmethod
    def _ensure_state(cls):
        if not hasattr(cls._local, 'paused'):
            cls._local.paused = False
        if not hasattr(cls._local, 'killed'):
            cls._local.killed = False

    @classmethod
    def is_paused(cls):
        cls._ensure_state()
        return cls._local.paused

    @classmethod
    def is_killed(cls):
        cls._ensure_state()
        return cls._local.killed

    @classmethod
    def set_paused(cls, value):
        cls._ensure_state()
        cls._local.paused = bool(value)

    @classmethod
    def set_killed(cls, value):
        cls._ensure_state()
        cls._local.killed = bool(value)


def pause():
    _TaskControl.set_paused(True)


def resume():
    _TaskControl.set_paused(False)


def kill():
    _TaskControl.set_killed(True)
'''


def generate_runtime_wrappers(
    include_checkpoint: bool = True,
    include_task_control: bool = True,
) -> str:
    """Build runtime wrappers in dependency-safe order."""
    parts = []
    if include_checkpoint:
        parts.append(generate_mobile_checkpoint_wrapper())
    if include_task_control:
        parts.append(generate_task_control_wrapper())
    return "\n".join(parts)


def instrument_for_task_control(
    func_code: str,
    checkpoint_state_vars: Optional[List[str]] = None
) -> Tuple[str, int, int]:
    """
    Instrument function code with pause/kill control checks
    
    Injects:
    - `if _TaskControl.is_killed(): return 'killed'` in loops
    - `while _TaskControl.is_paused(): time.sleep(0.1)` in loops
    
    Args:
        func_code: Original function source code
        checkpoint_state_vars: Optional list of checkpoint variable names for
            saving state on kill/pause
        
    Returns:
        Tuple of (instrumented_code, num_functions_instrumented, num_loops_instrumented)
    """
    try:
        tree = ast.parse(func_code)
        
        instrumenter = TaskControlInstrumenter(checkpoint_state_vars=checkpoint_state_vars)
        tree = instrumenter.visit(tree)
        
        ast.fix_missing_locations(tree)
        
        instrumented_code = ast.unparse(tree)
        
        return instrumented_code, instrumenter.functions_instrumented, instrumenter.loops_instrumented
        
    except Exception as e:
        logger.warning("[CodeInstrumenter] Task control instrumentation failed: %s", e)
        return func_code, 0, 0


def prepare_code_with_task_control(
    func_code: str,
    checkpoint_state_vars: Optional[List[str]] = None
) -> str:
    """
    Full pipeline: instrument code with task control checks and prepend wrapper
    
    Args:
        func_code: Original function source code
        checkpoint_state_vars: Optional list of checkpoint variable names
        
    Returns:
        Code with control wrapper prepended and control checks injected
    """
    if "__CROWDIO_TASK_CONTROL_WRAPPER__ = True" in func_code:
        return func_code

    instrumented, num_funcs, num_loops = instrument_for_task_control(
        func_code, checkpoint_state_vars=checkpoint_state_vars
    )
    logger.debug(
        "[CodeInstrumenter] Task control: instrumented %s functions, %s loops",
        num_funcs,
        num_loops,
    )

    wrapper = generate_runtime_wrappers(include_checkpoint=True, include_task_control=True)
    return wrapper + "\n" + instrumented
