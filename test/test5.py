import sys
import re
import ast
from collections import defaultdict, deque
import heapq
import time
import random
import math

# -----------------------------------------------------------------------------
# 赛题二：AI最短推理时延 - 完整解决方案
# 版本：混合策略（带健壮的计时器和超时中断）
# -----------------------------------------------------------------------------

def parse_input_line(line):
    # 此函数保持不变
    line = line.strip()
    match = re.match(r'(\w+)\((.*)\)', line, re.DOTALL)
    if not match: return None, None
    func_name, args_str = match.group(1), match.group(2)
    try:
        # 尝试使用ast.literal_eval进行更安全的解析
        return func_name, ast.literal_eval(args_str)
    except (ValueError, SyntaxError):
        # 作为备用方案，处理一些非标准格式
        try:
            return func_name, ast.literal_eval(args_str.replace('[', '(').replace(']', ')'))
        except: 
            return None, None

class Scheduler:
    # Scheduler 类的代码保持不变
    def __init__(self):
        self.soc_info = {}
        self.op_lib = {}
        self.available_tilings = defaultdict(list)

    def set_soc_info(self, cores, memory):
        core_id_counter = 0
        self.soc_info['core_map'] = {}
        temp_cores_by_type = defaultdict(list)
        for core_type, num in cores:
            for _ in range(num):
                self.soc_info['core_map'][core_id_counter] = core_type
                temp_cores_by_type[core_type].append(core_id_counter)
                core_id_counter += 1
        self.soc_info['cores_by_type'] = temp_cores_by_type
        self.soc_info['memory'] = dict(memory)

    def add_op_info(self, op_type, shape, tiling_id, sub_graph, run_detail, mem_detail):
        key = (op_type, shape, tiling_id)
        nodes = set(item[0] for item in run_detail)
        successors, in_degrees = defaultdict(list), defaultdict(int)
        for u, v in sub_graph:
            nodes.add(u); nodes.add(v)
            successors[u].append(v); in_degrees[v] += 1
        
        total_exec_time = sum(item[2] for item in run_detail)

        op_info = {
            'sub_graph_deps': sub_graph,
            'run_detail': {item[0]: {'core_type': item[1], 'exec_time': item[2]} for item in run_detail},
            'mem_detail': defaultdict(lambda: defaultdict(int)),
            'initial_nodes': sorted([n for n in nodes if in_degrees[n] == 0]),
            'terminal_nodes': sorted([n for n in nodes if not successors.get(n)]),
            'total_exec_time': total_exec_time
        }
        for nid, mtype, msize in mem_detail: op_info['mem_detail'][nid][mtype] += msize
        self.op_lib[key] = op_info
        if tiling_id not in self.available_tilings.get((op_type, shape), []):
            self.available_tilings[(op_type, shape)].append(tiling_id)
            self.available_tilings[(op_type, shape)].sort()

    def _build_full_dependency_graph(self, op_nodes_list, tiling_choices):
        self.nodes, self.successors, self.predecessors, self.predecessor_count, self.op_info_map = set(), defaultdict(list), defaultdict(list), defaultdict(int), {}
        self.op_nodes_list = op_nodes_list
        for op_id, op_type, shape in self.op_nodes_list:
            key = (op_type, shape, tiling_choices.get(op_id))
            if key not in self.op_lib: return False
            self.op_info_map[op_id] = self.op_lib[key]
            for sub_id in self.op_info_map[op_id]['run_detail']: self.nodes.add((op_id, sub_id))
        
        for op_id, info in self.op_info_map.items():
            for u, v in info['sub_graph_deps']:
                u_node, v_node = (op_id, u), (op_id, v)
                if u_node in self.nodes and v_node in self.nodes:
                    self.successors[u_node].append(v_node); self.predecessors[v_node].append(u_node); self.predecessor_count[v_node] += 1
        
        op_succs = defaultdict(list); [op_succs[u].append(v) for u, v in self.op_graph_info]
        for u_op, v_ops in op_succs.items():
            u_info = self.op_info_map.get(u_op)
            for v_op in v_ops:
                v_info = self.op_info_map.get(v_op)
                if u_info and v_info:
                    for u_sub in u_info['terminal_nodes']:
                        for v_sub in v_info['initial_nodes']:
                            u_node, v_node = (u_op, u_sub), (v_op, v_sub)
                            self.successors[u_node].append(v_node); self.predecessors[v_node].append(u_node); self.predecessor_count[v_node] += 1
        return True

    def _calculate_critical_path_priority(self):
        self.priorities = defaultdict(int)
        q = deque([n for n in self.nodes if not self.successors.get(n)])
        visited = set(q)
        while q:
            u_node = q.popleft()
            op_id, sub_id = u_node
            exec_time = self.op_info_map[op_id]['run_detail'][sub_id]['exec_time']
            max_p = max((self.priorities.get(v, 0) for v in self.successors.get(u_node, [])), default=0)
            self.priorities[u_node] = exec_time + max_p
            for pred in self.predecessors.get(u_node, []):
                if pred not in visited:
                    # 确保一个节点的所有后继都已被访问后，才将其加入队列
                    if all(succ_of_pred in visited for succ_of_pred in self.successors.get(pred, [])):
                        visited.add(pred)
                        q.append(pred)

    def schedule_attempt(self, op_graph, op_nodes, tiling_choices):
        # schedule_attempt 方法的核心调度逻辑保持不变
        self.op_graph_info = op_graph
        if not self._build_full_dependency_graph(op_nodes, tiling_choices):
            return float('inf'), None

        self._calculate_critical_path_priority()
        
        schedule_result, current_time, core_busy_until, memory_in_use = [], 0, defaultdict(int), defaultdict(int)
        running_nodes, ready_queue, finished_count, node_finish_times = [], [], 0, {}
        node_mem_detail_cache = {node: self.op_info_map[node[0]]['mem_detail'][node[1]] for node in self.nodes}
        
        local_predecessor_count = self.predecessor_count.copy()

        for node in self.nodes:
            if local_predecessor_count.get(node, 0) == 0:
                heapq.heappush(ready_queue, (-self.priorities[node], node))
        
        while finished_count < len(self.nodes):
            nodes_to_reschedule = []
            
            while ready_queue:
                priority, node = heapq.heappop(ready_queue)
                op_id, sub_id = node
                detail = self.op_info_map[op_id]['run_detail'][sub_id]
                core_type, mem_needed = detail['core_type'], node_mem_detail_cache[node]
                
                available_cores = [cid for cid in self.soc_info['cores_by_type'].get(core_type, []) if core_busy_until.get(cid, 0) <= current_time]
                mem_available = all(memory_in_use.get(m, 0) + s <= self.soc_info['memory'][m] for m, s in mem_needed.items())

                if available_cores and mem_available:
                    core_id = min(available_cores)
                    finish_time = current_time + detail['exec_time']
                    core_busy_until[core_id] = finish_time
                    for mt, ms in mem_needed.items(): memory_in_use[mt] = memory_in_use.get(mt, 0) + ms
                    heapq.heappush(running_nodes, (finish_time, node, core_id))
                    schedule_result.append([op_id, tiling_choices[op_id], sub_id, current_time, core_id])
                else:
                    nodes_to_reschedule.append((priority, node))
            
            for item in nodes_to_reschedule: heapq.heappush(ready_queue, item)
            
            if not running_nodes:
                if ready_queue: return float('inf'), None # 死锁检测
                else: break

            next_event_time = running_nodes[0][0]
            if not ready_queue or next_event_time > current_time:
                 current_time = next_event_time
            
            while running_nodes and running_nodes[0][0] <= current_time:
                finish_time, node, core_id = heapq.heappop(running_nodes)
                finished_count += 1
                node_finish_times[node] = finish_time
                
                # 内存释放逻辑简化（注意：真实赛题的内存释放逻辑更复杂，这里是简化版）
                mem_detail = node_mem_detail_cache[node]
                for mt, ms in mem_detail.items(): memory_in_use[mt] -= ms
                
                for succ in self.successors.get(node, []):
                    local_predecessor_count[succ] -= 1
                    if local_predecessor_count[succ] == 0:
                        heapq.heappush(ready_queue, (-self.priorities[succ], succ))
        
        if finished_count == len(self.nodes):
            makespan = max(node_finish_times.values()) if node_finish_times else 0
            return makespan, sorted(schedule_result)
        else:
            # 如果未能完成所有节点，则视为失败
            return float('inf'), None

def main():
    # 【修改点 1】设置计时器起点和总时间限制
    start_of_main = time.time()
    # 竞赛限制60秒，我们设定55秒作为安全阈值，为I/O等其他开销留出缓冲
    TIME_LIMIT_SECONDS = 55.0
    
    scheduler = Scheduler()
    content = sys.stdin.read()
    lines = content.strip().split('\n')
    
    op_info_lines, get_inference_line, set_soc_line = [], None, None
    for line in lines:
        if "AddOpInfo" in line: op_info_lines.append(line)
        elif "GetInferenceScheResult" in line: get_inference_line = line
        elif "SetSocInfo" in line: set_soc_line = line
    
    if set_soc_line:
        _, args = parse_input_line(set_soc_line)
        if args: scheduler.set_soc_info(*args)

    for op_line in op_info_lines:
        _, op_args = parse_input_line(op_line)
        if op_args: scheduler.add_op_info(*op_args)

    if get_inference_line:
        _, initial_args = parse_input_line(get_inference_line)
        if initial_args:
            op_graph, op_nodes_info = initial_args
            op_map = {op[0]: (op[1], op[2]) for op in op_nodes_info}
            
            # 初始化最优解记录
            best_schedule = None
            best_energy = float('inf')
            best_solution_choice = None

            # --- STAGE 1, Plan A: 贪心策略 ---
            sys.stderr.write("[Hybrid] Stage 1: 使用贪心策略生成初始解...\n")
            
            greedy_choice = {}
            for op_id, (op_type, shape) in op_map.items():
                available = scheduler.available_tilings.get((op_type, shape), [])
                if not available:
                    sys.stderr.write(f"[错误] 算子 {op_id} 没有任何可用的Tiling策略。程序退出。\n")
                    print("[]"); return

                best_tiling_id = -1; min_exec_time = float('inf')
                for tiling_id in available:
                    key = (op_type, shape, tiling_id)
                    op_info = scheduler.op_lib.get(key)
                    if op_info and op_info.get('total_exec_time', float('inf')) < min_exec_time:
                        min_exec_time = op_info['total_exec_time']
                        best_tiling_id = tiling_id
                greedy_choice[op_id] = best_tiling_id if best_tiling_id != -1 else available[0]

            initial_energy, initial_schedule = scheduler.schedule_attempt(op_graph, op_nodes_info, greedy_choice)

            # --- STAGE 1, Plan B: 如果贪心失败，则启动随机搜索作为后备 ---
            if initial_energy == float('inf'):
                sys.stderr.write("[警告] 贪心解不可行，启动随机搜索作为后备计划...\n")
                attempts = 0; max_attempts = 1000
                current_choice = {}
                while initial_energy == float('inf') and attempts < max_attempts:
                    # 【修改点 2】在随机搜索中加入超时检查
                    if (time.time() - start_of_main) > TIME_LIMIT_SECONDS:
                        sys.stderr.write("[超时] 寻找初始解阶段超时，输出空结果。\n")
                        print("[]"); return
                        
                    current_choice = {op_id: random.choice(scheduler.available_tilings[op_map[op_id]]) for op_id in op_map if op_map[op_id] in scheduler.available_tilings and scheduler.available_tilings[op_map[op_id]]}
                    initial_energy, initial_schedule = scheduler.schedule_attempt(op_graph, op_nodes_info, current_choice)
                    attempts += 1
                # 如果随机搜索成功，更新greedy_choice以反映当前的可行解
                if initial_energy < float('inf'):
                     greedy_choice = current_choice

            # --- 检查初始解是否找到 ---
            if initial_energy < float('inf'):
                best_energy, best_schedule, best_solution_choice = initial_energy, initial_schedule, greedy_choice
                sys.stderr.write(f"[Hybrid] 成功找到初始解，初始T: {best_energy:.0f}\n")
            else:
                sys.stderr.write(f"[错误] 在所有尝试后仍未能找到任何可行解。输出空结果。\n")
                print("[]"); return

            # --- STAGE 2: 以找到的可行解为起点，进行模拟退火精调 ---
            sys.stderr.write(f"[Hybrid] Stage 2: 以初始解为起点进行模拟退火...\n")
            
            current_solution, current_energy = best_solution_choice, best_energy
            
            initial_temp, final_temp, alpha = 1000.0, 1.0, 0.99
            temp = initial_temp
            
            # 【修改点 3】在主循环中严格遵守时间限制
            while temp > final_temp and (time.time() - start_of_main) < TIME_LIMIT_SECONDS:
                neighbor_solution = current_solution.copy()
                op_to_change = random.choice(list(op_map.keys()))
                op_type, shape = op_map[op_to_change]
                available = scheduler.available_tilings.get((op_type, shape), [])
                if len(available) <= 1:
                    continue # 如果没有其他选项，跳过此轮
                
                new_tiling = random.choice([t for t in available if t != current_solution[op_to_change]])
                neighbor_solution[op_to_change] = new_tiling

                neighbor_energy, neighbor_schedule = scheduler.schedule_attempt(op_graph, op_nodes_info, neighbor_solution)
                
                if neighbor_energy < float('inf'):
                    # 【修改点 4】修正最优解更新逻辑
                    # 检查是否找到了历史最优解
                    if neighbor_energy < best_energy:
                        best_energy = neighbor_energy
                        best_schedule = neighbor_schedule
                        best_solution_choice = neighbor_solution
                        sys.stderr.write(f"  [发现更优解] 新的T: {best_energy:.0f}, 温度: {temp:.2f}\n")
                    
                    # 根据Metropolis准则决定是否移动到新状态
                    energy_delta = neighbor_energy - current_energy
                    if energy_delta < 0 or (random.random() < math.exp(-energy_delta / temp)):
                        current_solution, current_energy = neighbor_solution, neighbor_energy
                
                temp *= alpha

            sys.stderr.write(f"\n[完成] 迭代结束或超时。找到的最优时间为: {best_energy if best_energy != float('inf') else 'N/A'}\n")
            # 【修改点 5】最终输出，如果 best_schedule 从未被成功赋值，则输出 "[]"
            print(str(best_schedule).replace(" ", "") if best_schedule else "[]")

if __name__ == "__main__":
    main()