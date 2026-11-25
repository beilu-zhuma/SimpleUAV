"""统一决策预算调度；适配器的一步返回False表示正常终止，异常原样交给批量恢复。"""

def run_decisions(limit, step):
    for iteration in range(limit):
        if step(iteration) is False:
            break
