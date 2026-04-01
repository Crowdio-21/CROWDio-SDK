
class CROWDioNamespace:
    """Namespace class for @CROWDio.task decorator style"""

    task = staticmethod(CROWDio_task)
    Constant = CROWDioConstant


CROWDio = CROWDioNamespace()