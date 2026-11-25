"""仅重试无运动副作用的模型HTTP请求；耗尽后交给任务级隔离。"""
import http.client
import json
import time
import urllib.error
import urllib.request

RECOVERY_POLICY = dict(attempts=3,retry_timeout_s=180,backoff_s=[5,15])


class ModelTransportFailure(RuntimeError):
    def __init__(self,error,attempts):
        self.attempts=attempts
        self.cause_type=type(error).__name__
        super().__init__(f'模型HTTP传输{attempts}次尝试耗尽：{type(error).__name__}: {error}')


def request_json(request,timeout,*,recovery=False,on_attempt=None,on_result=None):
    count=RECOVERY_POLICY['attempts'] if recovery else 1
    for attempt in range(count):
        if on_attempt:on_attempt()
        started=time.perf_counter() if on_result else None
        deadline=max(timeout,RECOVERY_POLICY['retry_timeout_s']) if attempt else timeout
        def report(result=None,error=None,retry=False):
            if on_result:
                on_result(dict(attempt=attempt,elapsed_s=time.perf_counter()-started,
                    timeout_s=deadline,success=error is None,error_type=type(error).__name__ if error else None,
                    http_status=getattr(error,'code',None),error=str(error)[:300] if error else None,
                    backoff_after_s=RECOVERY_POLICY['backoff_s'][attempt] if retry else 0,
                    usage=result.get('usage') if isinstance(result,dict) else None))
        try:
            with urllib.request.urlopen(request,timeout=deadline) as response:
                result=json.load(response)
        except urllib.error.HTTPError as error:
            retry=recovery and error.code in (500,502,503,504)
            report(error=error,retry=retry and attempt+1<count)
            if not retry:raise
            caught=error
            error.close()
        except (TimeoutError,urllib.error.URLError,ConnectionError,http.client.RemoteDisconnected) as error:
            report(error=error,retry=recovery and attempt+1<count)
            if not recovery:raise
            caught=error
        except BaseException as error:
            report(error=error)
            raise
        else:
            report(result=result)
            return result
        if attempt+1==count:raise ModelTransportFailure(caught,count) from caught
        time.sleep(RECOVERY_POLICY['backoff_s'][attempt])
