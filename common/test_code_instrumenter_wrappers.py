import re
from common.code_instrumenter import prepare_code_with_task_control

def test_wrapper_order_and_idempotency():
    src = '''def f(n):\n    s=0\n    for i in range(n):\n        s+=i\n    return s\n'''
    checkpoint_vars = ['i', 's']
    # First instrumentation
    out1 = prepare_code_with_task_control(src, checkpoint_vars)
    # Second instrumentation (should be idempotent)
    out2 = prepare_code_with_task_control(out1, checkpoint_vars)

    # Check marker
    assert '__CROWDIO_TASK_CONTROL_WRAPPER__ = True' in out1
    assert out1 == out2, 'Instrumentation is not idempotent!'

    # Check order: checkpoint wrapper before task control wrapper
    idx_ckpt = out1.find('class _MobileCheckpointState')
    idx_ctrl = out1.find('class _TaskControl')
    assert idx_ckpt != -1 and idx_ctrl != -1, 'Wrappers missing!'
    assert idx_ckpt < idx_ctrl, 'Checkpoint wrapper must come before task control wrapper!'

    # Check for new task-scoped control checks
    assert '_TaskControl.is_killed()' in out1
    assert '_TaskControl.is_paused()' in out1

    # No global paused/killed
    assert 'global paused' not in out1
    assert 'global killed' not in out1

    print('All wrapper order/idempotency checks passed.')

if __name__ == '__main__':
    test_wrapper_order_and_idempotency()
